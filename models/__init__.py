from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PegasusTokenizer, PegasusTokenizerFast, PegasusModel, PegasusForConditionalGeneration, PegasusConfig

from transformers.models.pegasus.modeling_pegasus import PegasusAttention, PegasusDecoderLayer, PegasusAttention, shift_tokens_right
from .gat import GraphEncoder
from .pegasus import PegasusSuperDecoder, PegasusSuperDecoderLayer


class Model(PreTrainedModel):
    def __init__(self, graph=True, encoders=5, decoders=3, shared_head=False, pretrained=True, vocab_size=96103):
        # Pegasus model
        if pretrained:
            pegasus_model = PegasusForConditionalGeneration.from_pretrained("sshleifer/distill-pegasus-cnn-16-4")
        else:
            configuration = PegasusConfig(d_model=768, vocab_size=vocab_size,
                                          encoder_layers=encoders, decoder_layers=decoders,
                                          encoder_attention_heads=12, decoder_attention_heads=12,
                                          decoder_ffn_dim=3072, encoder_ffn_dim=3072)
            pegasus_model = PegasusForConditionalGeneration(configuration)
        super().__init__(config=pegasus_model.config)
        self.config.pretrained = pretrained

        # pegasus encoder
        # self.prepare_inputs_for_generation = pegasus_model.prepare_inputs_for_generation
        self.shared = pegasus_model.model.shared
        self.config.encoder_layers = encoders
        pegasus_model.model.encoder.layers = pegasus_model.model.encoder.layers[:encoders]
        self.encoder = pegasus_model.model.encoder

        # gat model
        self.graph = graph
        if self.graph:
            self.gat_model = GraphEncoder(num_of_layers=3, num_heads_per_layer=[8, 8, 8], num_features_per_layer=[128, 256, 512, self.config.d_model])

        # pegasus decoder
        self.config.decoder_layers = decoders
        pegasus_model.model.decoder.layers = pegasus_model.model.decoder.layers[:decoders]
        self.decoder = pegasus_model.model.decoder
        if self.graph:
            self.decoder.layers = nn.ModuleList([PegasusSuperDecoderLayer(self.config, decoder_layer) for decoder_layer in self.decoder.layers])
            self.decoder = PegasusSuperDecoder(self.config, self.decoder)

        # resulting
        if shared_head:
            self.lm_head = pegasus_model.lm_head
        else:
            self.lm_head = nn.Linear(pegasus_model.lm_head.in_features, pegasus_model.lm_head.out_features, bias=False)
            self.lm_head.weight = copy.deepcopy(pegasus_model.lm_head.weight)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and argument != 'input_nodes_embeddings' and argument != 'input_edges'
            }
            model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **encoder_kwargs)
            if self.graph:
                gat_hidden_states, gat_attention_mask = self.gat_model(in_nodes_features_batch=model_kwargs["input_nodes_embeddings"], topology_batch=model_kwargs["input_edges"])
                model_kwargs["gat_outputs"] = gat_hidden_states
                model_kwargs["gat_attention_mask"] = gat_attention_mask
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs = None,
        gat_outputs = None,
        gat_attention_mask: torch.LongTensor = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if gat_attention_mask is not None:
            model_kwargs["gat_attention_mask"] = gat_attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

            if gat_outputs is not None:
                model_kwargs["gat_outputs"] = gat_outputs.index_select(0, expanded_return_idx.to(gat_outputs.device))
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        gat_outputs=None,
        gat_attention_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # print('\n+++ prepare_inputs_for_generation')
        # print('encoder_outputs ', encoder_outputs.last_hidden_state.shape if encoder_outputs is not None else None)
        # print('attention_mask ', attention_mask.shape if attention_mask is not None else None)
        # print('gat_outputs ', gat_outputs.shape if gat_outputs is not None else None)
        # print('gat_attention_mask ', gat_attention_mask.shape if gat_attention_mask is not None else None)

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "gat_outputs": gat_outputs,
            "gat_attention_mask": gat_attention_mask,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @torch.no_grad()
    def predict(self, input_ids=None, attention_mask=None, input_nodes_embeddings=None, input_edges=None):
        if self.graph:
            return self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 input_nodes_embeddings=input_nodes_embeddings, input_edges=input_edges,
                                 do_sample=True, repetition_penalty=2.0, temperature=1.2, top_k=50,
                                 min_length=40, max_length=90, num_beams=2, early_stopping=True)
        else:
            return self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 do_sample=True, repetition_penalty=2.0, temperature=1.2, top_k=50,
                                 min_length=40, max_length=90, num_beams=2, early_stopping=True)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                encoder_outputs=None, output_attentions=None, output_hidden_states=None,
                input_nodes_embeddings=None, input_edges=None, labels=None, gat_outputs=None, gat_hidden_states=None, gat_attention_mask=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if gat_outputs is not None:
            gat_hidden_states = gat_outputs

        # print('\n+++ model forward')
        # print('encoder')
        # print('input_ids ', input_ids.shape if input_ids is not None else None)
        # print('encoder_outputs ', encoder_outputs.last_hidden_state.shape if encoder_outputs is not None else None)
        # print('attention_mask ', attention_mask.shape if attention_mask is not None else None)
        #
        # print('gat')
        # print('input_nodes_embeddings', [i.shape for i in input_nodes_embeddings] if input_nodes_embeddings is not None else None)
        # print('input_edges', [i.shape for i in input_edges] if input_edges is not None else None)
        # print('gat_outputs ', type(gat_outputs), gat_outputs.shape if gat_outputs is not None else None)
        # print('gat_hidden_states ', gat_hidden_states.shape if gat_hidden_states is not None else None)
        # print('gat_attention_mask ', gat_attention_mask.shape if gat_attention_mask is not None else None)

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states)

        if self.graph:
            if (input_nodes_embeddings is None or input_edges is None) and (gat_hidden_states is None or gat_attention_mask is None):
                raise Exception('input_nodes_embeddings & input_edges is None or gat_hidden_states & gat_attention_mask is None')

            if gat_hidden_states is None or gat_attention_mask is None:
                gat_hidden_states, gat_attention_mask = self.gat_model(input_nodes_embeddings, input_edges)

            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs[0],
                                           encoder_attention_mask=attention_mask,
                                           gat_hidden_states=gat_hidden_states,
                                           gat_attention_mask=gat_attention_mask,
                                           input_ids=decoder_input_ids,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states)
        else:
            decoder_outputs = self.decoder(encoder_hidden_states=encoder_outputs[0],
                                           encoder_attention_mask=attention_mask,
                                           input_ids=decoder_input_ids,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states)

        # lm_logits = self.lm_head(decoder_outputs[0].to('cuda:1')).to('cuda')
        lm_logits = self.lm_head(decoder_outputs[0])

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


def get_model(graph=True, encoders=4, decoders=3, shared_head=False, pretrained=True):
    pegasus_tokenizer = PegasusTokenizerFast.from_pretrained("sshleifer/distill-pegasus-cnn-16-4")

    model = Model(shared_head=shared_head, graph=graph, encoders=encoders, decoders=decoders, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    if graph:
        for param in model.gat_model.parameters():
            param.requires_grad = True
    for param in model.shared.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return pegasus_tokenizer, model
