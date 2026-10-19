"""Transport-safe native context overflow diagnostics (standard library only)."""


class NativeContextBudgetError(ValueError):
    def __init__(self, input_tokens: int, output_tokens: int, limit: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.limit = limit
        super().__init__(
            f"Native context window exceeded: input {input_tokens} + "
            f"output reserve {output_tokens} > active slot {limit}; "
            "split the task; nothing was truncated"
        )
