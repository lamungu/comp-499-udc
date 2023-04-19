
#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence dialogue system
The system employs a Transformer Language Model. 

To run this recipe, do the following:
> python train.py train.yaml

With the default hyperparameters, the system employs an GRU encoder and decoder.

The neural network is trained with the negative-log likelihood objective and
characters are used as basic tokens for both english and ignotush.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)



# Brain class for speech recognition training
class Dialogue(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.tensor
            Log-probabilities predicted by the decoder.
            
        At validation/test time, it returns the predicted tokens as well.
        """
        
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device) # todo
        
        # Unpack the encoded context 
        enc_context, context_lens = batch.context_tokens
        
        # Context embeddings
        context_embeddings = self.modules.embeddings(enc_context)
        
        # Running the encoder
        encoded_signal, _ = self.modules.encoder(context_embeddings)
    
        # Unpacking the responses (with bos tokens)
        enc_response, response_lens = batch.response_encoded_tokens_bos
        
        # Response Embeddings
        response_embeddings = self.modules.embeddings(enc_response)

        # Running the decoder
        decoder_outputs, _ = self.modules.decoder(response_embeddings, encoded_signal, context_lens)
        
        # Compute logits (i.e., apply final linear transformation)
        # Your code here. Aim for 1 line
        logits = self.modules.seq_lin(decoder_outputs)
        
        # Compute log posteriors
        # Your code here. Aim for 1 line
        predictions = self.hparams.log_softmax(logits)
        
        if stage == sb.Stage.TEST:
            
            # Getting some predictions.
            hyps = predictions.argmax(-1)
            
            # getting the first index where the prediction is eos_index
            stop_indexes = (hyps == self.hparams.eos_index).int()
            stop_indexes = stop_indexes.argmax(dim=1)
            
            # Converting hyps from indexes to chars
            hyp_lst = []
            for hyp, stop_ind in zip(hyps, stop_indexes):
                # in some cases the eos is not observed (e.g, for the last sentence
                # in the batch)
                if stop_ind == 0:
                    stop_ind = -1
                # Stopping when eos is observed
                hyp = hyp[0:stop_ind]

                # From index to character
                hyp_lst.append(self.hparams.tokenizer.decode(hyp.tolist()))

            return predictions, hyp_lst
        
        return predictions
        

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        
        Arguments
        ---------
        predictions : torch.tTensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Unpack the response labels (with <eos>)
        enc_response_eos, english_lens = batch.response_encoded_tokens_eos
        
        # Reading the predictions
        if stage == sb.Stage.TEST:
          predictions, hyp_lst = predictions
          
          for inp, label, hyp in zip(batch.context, batch.response, hyp_lst):
              
              print("INP: " + inp)
              print("REF: " + label)
              print("HYP: " + hyp)
              print('--------')
        
        # Computing the nll_loss
        # Your code here. Aim for 1 line
        loss = sb.nnet.losses.nll_loss(predictions, enc_response_eos, english_lens)   
        
        return loss

            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

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
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                },
            )
            

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    """


    tokenizer = hparams["tokenizer"]
    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]

    # Define text processing pipeline. We start from the raw text and then
    # split it into characters. The tokens with BOS are used for feeding
    # the decoder during training (right shifr), the tokens with EOS 
    # are used for computing the cost function.
    @sb.utils.data_pipeline.takes("context", "response")
    @sb.utils.data_pipeline.provides(
        context
        response
        response_encoded_tokens
        response_encoded_masks
        response_encoded_tokens_bos
        response_encoded_masks_bos
        )
    def text_pipeline(context, response):
        """Processes the transcriptions to generate proper labels"""
        
        yield context
        yield response
        # Set it in a way GPT2 can read and learn from it
        tokens = tokenizer.encode('<|context|> %s <|response|> %s <|endoftext|>' % (context, response))
        response_encoded_tokens, response_encoded_masks = tokens['input_ids'], tokens['attention_masks']
        yield response_encoded_tokens
        yield response_encoded_masks

        tokens_bos = tokenizer.encode('<|context|> %s <|response|>')
        response_encoded_tokens_bos, response_encoded_masks_bos = tokens_bos['input_ids'], tokens_bos['attention_masks']
        yield response_encoded_tokens_bos
        yield response_encoded_masks_bos
      
        

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

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    translate_brain.fit(
        translate_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = translate_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
