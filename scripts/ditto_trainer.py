"""DITTO-specific behavior layered on TRL's pinned DPOTrainer."""

from transformers import TrainerCallback
from trl import DPOTrainer

from scripts.dataset_utils import DITTODataCollator


class ResampleCallback(TrainerCallback):
    """Refresh online comparison samples at the configured optimizer steps."""

    def __init__(self, collator, resample_rate: int):
        self.collator = collator
        self.resample_rate = resample_rate
        self.last_step_num = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._resample_if_due(state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self._resample_if_due(state.global_step)

    def _resample_if_due(self, step: int):
        step = int(step)
        if self.last_step_num == step:
            return
        if step % self.resample_rate == 0:
            self.collator.resample(step=step)
        self.last_step_num = step


class DITTOTrainer(DPOTrainer):
    """TRL DPOTrainer with DITTO's online comparison-data collator."""

    def __init__(
        self,
        model=None,
        *,
        bootstrap_count: int = 10,
        resample_rate: int = 10,
        generation_max_new_tokens: int = 1024,
        generation_temperature: float = 1.0,
        generation_batch_size: int = 1,
        **trainer_kwargs,
    ):
        if bootstrap_count <= 0:
            raise ValueError("bootstrap_count must be a positive integer.")
        if resample_rate <= 0:
            raise ValueError("resample_rate must be a positive integer.")
        supplied_collator = trainer_kwargs.pop("data_collator", None)
        if supplied_collator is not None:
            raise ValueError("DITTOTrainer constructs its online data collator internally.")

        args = trainer_kwargs.get("args")
        tokenizer = trainer_kwargs.get("tokenizer")
        train_dataset = trainer_kwargs.get("train_dataset")
        if args is None or tokenizer is None or train_dataset is None:
            raise ValueError("DITTOTrainer requires args, tokenizer, and train_dataset.")
        if isinstance(model, str) or model is None:
            raise ValueError("DITTOTrainer requires an already-instantiated model.")

        # Unlike SFT's tokenized dataset, the online collator consumes the raw
        # prompt, chosen, and example_id columns.
        args.remove_unused_columns = False

        max_length = trainer_kwargs.get("max_length") or 512
        max_prompt_length = trainer_kwargs.get("max_prompt_length") or 128
        collator = DITTODataCollator(
            pad_token_id=tokenizer.pad_token_id,
            model=model,
            tokenizer=tokenizer,
            bootstrap_count=bootstrap_count,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_temperature=generation_temperature,
            generation_batch_size=generation_batch_size,
            train_dataset=train_dataset,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            label_pad_token_id=trainer_kwargs.get("label_pad_token_id", -100),
            frac_expert=args.frac_expert,
            frac_intermodel=args.frac_intermodel,
            frac_replay=args.frac_replay,
            rescale_batch=args.rescale_batch,
        )

        callbacks = list(trainer_kwargs.pop("callbacks", None) or [])
        callbacks.append(ResampleCallback(collator, resample_rate))
        super().__init__(
            model=model,
            data_collator=collator,
            callbacks=callbacks,
            **trainer_kwargs,
        )

    def tokenize_row(self, feature, model=None):
        """Keep raw rows; the online collator tokenizes generated comparisons."""

        return feature
