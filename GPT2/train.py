
#!/usr/bin/env/python3
"""Recipe for training an End-to-End dialogue system
The system employs a GPT-2 based Transformer Language Model. 

To run this recipe, do the following:
> python train_GPT2.py GPT2.yaml
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from itertools import chain
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import batch_pad_right
import torch.nn.functional as F

from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)

# Brain class for speech recognition training
class Dialogue(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on Transformer flow"""
        
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device) # tod
        
        # Get the inputs and token types
        input_ids, _ = batch.input_ids
        token_type_ids, _ = batch.token_type_ids

        # Running the model
        outputs = self.modules.llm_model(
            input_ids,
            token_type_ids
        ).logits

        #  apply softmax if necessary
        predictions = self.hparams.log_softmax(outputs)

        if stage == sb.Stage.TEST:
            # This should do beamsearch, but for now we'll do
            # Greedy search
            predicted_tokens = predictions.argmax(-1)
            
            # Getting the first index in the sequence where the prediction is eos_index
            eos_indices = (predicted_tokens == self.eos_token).int()
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
        """Computes the loss given the predicted and targeted outputs.
        """

        # Get the labels
        response_labels, response_lengths = batch.lm_labels
        input_ids, input_lengths = batch.input_ids
        

        # Reading the predictions
        if stage == sb.Stage.TEST:
          predictions, outputs = predictions
          decoded_outputs = [self.hparams.tokenizer.decode(o.tolist()) for o in outputs]
          outputs, _ = batch_pad_right(outputs);
          ids = batch.utt_id
       
          # Get Embeddings for metrics
          label_embs = self.modules.llm_model(
              input_ids,
              torch.zeros(response_labels.shape, dtype=torch.long, device=self.device),
              output_hidden_states=True,
          ).hidden_states
          output_embs = self.modules.llm_model(
              outputs,
              torch.zeros(outputs.shape, dtype=torch.long, device=self.device),
              output_hidden_states=True,
          ).hidden_states

          # Take the first layer to compare the metrics
          label_embs = label_embs[0]
          output_embs = output_embs[0]

          # Embedding Average Score
          label_avg = torch.mean(label_embs, dim=1)
          output_avg = torch.mean(output_embs, dim=1)
          self.eas_metric.append(ids, output_avg, label_avg)
          
          # Vector Extrema Score
          label_max, _ = torch.max(label_embs, dim=1)
          output_max, _ = torch.max(output_embs, dim=1)
          self.ves_metric.append(ids, output_max, label_max)

          # Add word error rate (WER) metric for the predicted words
          predicted_words = [o.split(' ') for o in decoded_outputs]
          target_words = [r.split(" ") for r in batch.response]
          self.wer_metric.append(ids, predicted_words, target_words)
          self.bleu_metric.append(ids, predicted_words, [target_words])

          with open(self.hparams.gen_file, "a") as w:
            for id, input, response, output in zip(batch.utt_id, batch.context, batch.response, decoded_outputs):
                
                # generate text until the output length (which includes the context length) reaches 50
                # input_ids = self.hparams.tokenizer.encode(input, return_tensors='pt')
                # input_ids = input_ids.to(self.device)
                # output = self.modules.llm_model.generate(input_ids, max_new_tokens=50)

                w.write("INP: %s \n" % input)      
                w.write("REF: %s \n" % response)
                w.write("OUT: %s \n" % output)
                w.write('--------\n')
          
        # Calculate the loss
        loss = self.hparams.compute_cost(predictions, response_labels, response_lengths)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        bos_token, eos_token, ctx_token, rep_token =  self.hparams.tokenizer.convert_tokens_to_ids(hparams["special_tokens"])

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.ctx_token = ctx_token
        self.rep_token = rep_token

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


        # Perform end-of-iteration things, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                },
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
    
    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)

# Here, we add special tokens to the tokenizer
# In order to specify and delineate the context
# from the response inside the GPT2 input.
def add_special_tokens_(
    model,
    tokenizer,
    attr_to_special_token,
) -> None:
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        attr_to_special_token  # type: ignore
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
 

def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    
    # convert special tokens to their ids
    bos_token, eos_token, ctx_token, rep_token =  tokenizer.convert_tokens_to_ids(hparams["special_tokens"])

    @sb.utils.data_pipeline.takes("context", "response")
    @sb.utils.data_pipeline.provides(
        "context",
        "response",
        "input_ids",
        "token_type_ids",
        "lm_labels"
    )
    def input_and_token_type_pipeline(context, response):
        """Processes the transcriptions to generate proper labels"""
        yield context
        yield response

        # Encode the context and responses
        context_tokens = tokenizer.encode(context)
        response_tokens = tokenizer.encode(response)
        

        # Add the special tokens to the beginning of their sentences
        context_tokens = [ctx_token] + context_tokens
        response_tokens = [rep_token] + response_tokens

        # add the eos token if the parameter is on
        if hparams["with_eos"]:
            response_tokens = response_tokens + [eos_token]

        # concatenate the tokens to create the input sequence 
        input_ids = context_tokens + response_tokens
        input_ids = torch.LongTensor(input_ids)
        yield input_ids

        # Create the token type ids based on the lists above
        context_type_ids = [ctx_token]*len(context_tokens)
        reply_type_ids = [rep_token]*len(response_tokens)
        token_type_ids = context_type_ids + reply_type_ids
        token_type_ids = torch.LongTensor(token_type_ids)
        yield token_type_ids

        # Finally, create the ground truth label that will be used to
        # evaluate the reply generated by the LLM. -100 is a special
        # token that is ignored during the loss computation; the
        # idea is to mask everything except the reply produced.
        lm_labels = ([-100] * len(context_tokens)) + [-100] + response_tokens[1:]
        lm_labels = torch.LongTensor(lm_labels)
        yield lm_labels

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
            dynamic_items=[input_and_token_type_pipeline],
            output_keys=[
                "utt_id",
                "context",
                "response",
                "input_ids",
                "token_type_ids",
                "lm_labels"
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = True
        
    return datasets


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Add special tokens to the tokenizer and resize model embedding
    add_special_tokens_(hparams['llm_model'].model, hparams['tokenizer'], hparams["attr_to_special_tokens"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, hparams['tokenizer'])

    # Trainer initialization
    dialogue_brain = Dialogue(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training sequence
    dialogue_brain.fit(
        dialogue_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = dialogue_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
