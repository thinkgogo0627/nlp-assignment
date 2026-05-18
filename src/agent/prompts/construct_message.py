from .prompt import PROMPT_REGISTRY


class Message:
    def __init__(self, system_prompt: str, few_shot_examples: list[dict] = None):
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples

    def get_system_prompt(self):
        return {"role": "system", "content": self.system_prompt}

    def get_user_message(self, content: str):
        return {"role": "user", "content": content}

    def get_assistant_message(self, content: str):
        return {"role": "assistant", "content": content}

    def get_few_shot_examples(self):
        if self.few_shot_examples is None:
            return []

        few_shot_prompts = []
        for example in self.few_shot_examples:
            for shot in example:
                few_shot_prompts.append({
                    "role": shot["role"],
                    "content": shot["content"],
                })

        return few_shot_prompts

    def get_message(self, trajectories: list[str]):
        messages = [self.get_system_prompt()]
        messages.extend(self.get_few_shot_examples())

        # We keep the basic alternating user/assistant format stable so that
        # students can focus on multi-agent design rather than LangChain syntax.
        for idx, trajectory in enumerate(trajectories):
            if idx % 2 == 0:
                messages.append(self.get_user_message(trajectory))
            else:
                messages.append(self.get_assistant_message(trajectory))

        return messages


def construct_message(
    prompt_type: str,
    trajectories: list[str],
    few_shot_examples: list[dict] = None,
    system_prompt: str = None,
):
    """
    Construct messages for 2WikiMultiHopQA agents.

    Students may define their own agent roles in prompt.py and register them in
    PROMPT_REGISTRY. For example:

        PROMPT_REGISTRY = {
            "single_agent": ...,
            "decomposer": ...,
            "agent_1": ...,
            "agent_2": ...,
            "agent_3": ...,
        }

    Required baseline prompt:
    - single_agent

    Required multi-agent prompt:
    - decomposer

    Student-designed prompts:
    - agent_1, agent_2, agent_3, ... or any custom role names

    TODO for students:
    1. Add your own agent prompts to PROMPT_REGISTRY in prompt.py.
    2. Decide whether each agent should receive the same context or a modified
       context from previous agents.
    3. Optionally customize message formatting below.
    """
    if system_prompt is None:
        if prompt_type not in PROMPT_REGISTRY:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. "
                "Please add this prompt type to PROMPT_REGISTRY in prompt.py."
            )
        system_prompt = PROMPT_REGISTRY[prompt_type]

    # TODO(optional): Students may modify this block if they want different
    # message formatting for different agents. For example, a verifier could
    # receive evidence in a stricter format, while a final answer agent could
    # receive only selected evidence.
    message = Message(
        system_prompt=system_prompt,
        few_shot_examples=few_shot_examples,
    )

    return message.get_message(trajectories)
