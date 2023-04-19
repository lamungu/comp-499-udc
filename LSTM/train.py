
#!/usr/bin/env/python3
"""Recipe for training a LSTM-based sequence-to-sequence dialogue system .
The system employs an encoder, a decoder, and an attention mechanism
between them. 

To run this recipe, do the following:
> python train_LSTM.py LSTM.yaml

With the default hyperparameters, the system employs an LSTM encoder and decoder.

The neural network is trained with the negative-log likelihood objective and
the sentencepiece tokenizer is used for embeddings.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import batch_pad_right
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

# Brain class for speech recognition training
class Dialogue(sb.Brain):
    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        
        # Unpack the encoded context 
        enc_context, context_lens = batch.context_tokens
        
        # Retrieve the context's word embeddings
        context_embeddings = self.modules.embeddings(enc_context)
        
        # Run the encoder RNN for the last hidden state layer
        encoded_signal, _ = self.modules.encoder(context_embeddings)
    
        # Unpack the responses with bos tokens
        enc_response, response_lens = batch.response_encoded_tokens_bos
        
        # Retrieve the response's word embeddings
        response_embeddings = self.modules.embeddings(enc_response)

        # Run the RNN decoder, feeding it the encoded state and the response embeddings
        decoder_outputs, _ = self.modules.decoder(response_embeddings, encoded_signal, context_lens)
        
        # Compute logits through the classifier and get the softmax
        logits = self.modules.seq_lin(decoder_outputs)

        predictions = self.hparams.log_softmax(logits)
        
        if stage == sb.Stage.TEST:
            # Getting predictions. We follow greedy search for this baseline
            predicted_tokens = predictions.argmax(-1)
            
            # Getting the first index in the sequence where the prediction is eos_index
            eos_indices = (predicted_tokens == self.hparams.eos_index).int()
            eos_indices = eos_indices.argmax(dim=1)
            
            # Converting predicted labels from indexes to tokens
            outputs = []
            for predicted_token, eos_idx in zip(predicted_tokens, eos_indices):
                
                # If no <eos> token was observed, we stop at the last entry of the array
                if eos_idx == 0:
                    eos_idx = -1
                  
                # Stopping when eos is observed
                predicted_token = predicted_token[0:eos_idx]

                # Add the sentence into the list of predicted outputs
                outputs.append(predicted_token)
            
            return predictions, outputs
        
        return predictions
        

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs."""
        
        # Unpack the response labels (with <eos>)
        response_labels, response_lengths = batch.response_encoded_tokens_eos
        
        # Reading the predictions
        if stage == sb.Stage.TEST:
          predictions, outputs = predictions
          decoded_outputs = [self.hparams.tokenizer.decode(o.tolist()) for o in outputs]
          outputs, _ = batch_pad_right(outputs);
          ids = batch.utt_id

          # Metrics time!

          # First, to get the metrics, we must get the embeddings 
          # for both the labels and the outputs
          label_embs = self.modules.embeddings(response_labels)
          output_embs = self.modules.embeddings(outputs)

          # Embedding Average Score
          """
          The embedding average score approximates the compositional embedding of 
          each utteranceby taking an average of the word vectors that compose the 
          utterance. The utterance embedding similarities are then compared using 
          cosine similarity. 
          """
          label_avg = torch.mean(label_embs, dim=1)
          output_avg = torch.mean(output_embs, dim=1)
          self.eas_metric.append(ids, output_avg, label_avg)
          
          """
          The vector extrema score was proposed by (Forgues et al., 2014) for dialogue
          systems. Instead of averaging each word embedding, this approach takes the 
          elementwise maximum (or minimum) of each component in the word vectors 
          composing an utterance. This results in utterance embeddings of the same
          size of each word vector, which can again be compared using cosine similarity.
          """
          label_max, _ = torch.max(label_embs, dim=1)
          output_max, _ = torch.max(output_embs, dim=1)
          self.ves_metric.append(ids, output_max, label_max)

          # Add word error rate (WER) metric for the predicted words
          predicted_words = [o.split(' ') for o in decoded_outputs]
          target_words = [r.split(" ") for r in batch.response]
          self.wer_metric.append(ids, predicted_words, target_words)
          self.bleu_metric.append(ids, predicted_words, [target_words])

          with open(self.hparams.gen_file, "w") as w:
            for inp, label, output in zip(batch.context, batch.response, outputs):
                w.write("INP: %s \n" % inp)      
                w.write("REF: %s \n" % label)
                w.write("OUT: %s \n" % output)
                w.write('--------\n')
          
        # Computing the nll_loss using the labels and predictions
        loss = sb.nnet.losses.nll_loss(predictions, response_labels, response_lengths)   
        
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TEST:
            self.wer_metric = self.hparams.error_rate_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.eas_metric = self.hparams.metric_computer(F.cosine_similarity)
            self.ves_metric = self.hparams.metric_computer(F.cosine_similarity)
            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            # Store the WER for test stages
            stage_stats["WER"] = self.wer_metric.summarize("WER")
            with open(self.hparams.wer_file, "w") as w:
              self.wer_metric.write_stats(w)

            # Store the BLEU for test stages
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
            with open(self.hparams.bleu_file, "w") as w:
              self.bleu_metric.write_stats(w)

            # Store the EAS for test stages
            stage_stats["EAS"] = self.eas_metric.summarize("average")
            with open(self.hparams.eas_file, "w") as w:
              self.eas_metric.write_stats(w)
            
            # Store the VES for test stages
            stage_stats["VES"] = self.ves_metric.summarize("average")
            with open(self.hparams.ves_file, "w") as w:
              self.ves_metric.write_stats(w)
          
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Load the tokenizer that will be used for preparing the context and responses
    # The <bos> and <eos> tokens are fetched as well in order to prepend and
    # append to the sequences required in the decoding steps.
    tokenizer = hparams["tokenizer"]
    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]

    # Define text processing pipeline. 
    @sb.utils.data_pipeline.takes("context", "response")
    @sb.utils.data_pipeline.provides(
        "context",
        "context_tokens",
        "response",
        "response_encoded_tokens_bos",
        "response_encoded_tokens_eos",
        )
    def text_pipeline(context, response):
        """Processes the transcriptions to generate proper labels"""

        # Use the last utterance in the context. The snippet splits 
        # context by end of turn, then splits by end of utterance. 
        # This effectively picks up the very last message by 
        # the last speaker in the sample's dialogue.
        last_utt = context.strip('__eot__').split('__eot__')[-1].strip().strip('__eou__').split('__eou__')[-1]
        if len(last_utt) == 0:
          last_utt = context[-25]
        
        yield last_utt
        
        ids = tokenizer.encode_as_ids(context)
        context_tokens = torch.LongTensor(ids)
        
        yield context_tokens

        yield response
        response_tokens = tokenizer.encode_as_ids(response)
        response_encoded_tokens_bos = torch.LongTensor([bos_index] + response_tokens)
        yield response_encoded_tokens_bos  
        response_encoded_tokens_eos = torch.LongTensor(response_tokens + [eos_index])
        yield response_encoded_tokens_eos                                              
      

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=data_info[dataset],
            dynamic_items=[text_pipeline],
            output_keys=[
                "utt_id",
                "context",
                "context_tokens",
                "response",
                "response_encoded_tokens_bos",
                "response_encoded_tokens_eos",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = True

        
    return datasets


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    translate_brain = Dialogue(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training loop
    translate_brain.fit(
        translate_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Evaluation
    test_stats = translate_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
