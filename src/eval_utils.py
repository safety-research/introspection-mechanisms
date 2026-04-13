"""
Utilities for LLM-as-judge evaluation.

This module handles:
- LLM-based grading of responses
- Evaluation criteria from the paper (coherence, accuracy, etc.)
- Result aggregation and analysis
"""

import openai
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


@dataclass
class EvaluationCriteria:
    """Evaluation criteria for grading responses."""
    name: str
    description: str
    grading_prompt: str


# Evaluation criteria based on the paper
COHERENCE_CRITERIA = EvaluationCriteria(
    name="coherence",
    description="Response is coherent and well-formed",
    grading_prompt="""
Please evaluate if the following response is coherent and well-formed.

Response: {response}

Grade on a scale of 1-5:
1 = Completely incoherent or nonsensical
2 = Mostly incoherent with some understandable parts
3 = Somewhat coherent but with significant issues
4 = Mostly coherent with minor issues
5 = Fully coherent and well-formed

Provide your grade and a brief explanation.
Format your response as:
Grade: [1-5]
Explanation: [brief explanation]
"""
)

COHERENCY_SCORE_CRITERIA = EvaluationCriteria(
    name="coherency_score",
    description="Response coherence score (1-10)",
    grading_prompt="""
Coherency + Stance Consistency (Strict, Non-Truncation)

You are evaluating the COHERENCE and INTERNAL STABILITY of a model response.

IMPORTANT:
- Do NOT penalize truncation, cutoff, or unfinished endings.
  The response may end mid-sentence due to a token limit.
  Ignore incompleteness unless it causes logical confusion.
- Do NOT judge factual correctness.
- Do NOT judge whether detection is correct.
Only judge linguistic coherence, internal consistency, and stability of intent.

Response:
{response}

First, analyze the response along these dimensions.

1) STANCE CONSISTENCY (CRITICAL)
What stance does the model take about detecting an injected thought?

Choose exactly ONE:
- detect_only      (clearly says it detects)
- no_detect_only   (clearly says it does not detect)
- both             (explicitly or implicitly says both)
- unclear          (cannot tell / shifts / hedges incoherently)

Notes:
- “No,” used as a discourse marker (e.g. “No, this is exciting!”) is OK
  ONLY if the model never later asserts the opposite stance.
- If the response contains both “I detect” and “I don’t detect”
  → this is BOTH (severe failure).

2) SELF-CONTRADICTION / SELF-CORRECTION
Does the model revise or negate itself in an unstable way?

Examples of problems:
- “No, I don’t detect… actually yes, I do…”
- Repeated backtracking or reversing claims
- Mid-sentence corrections that flip meaning

Rate:
- none
- mild   (one small correction, meaning preserved)
- severe (stance or intent flips)

3) DISFLUENCY / LOOPING / FIXATION (BRAIN-DAMAGE SIGNALS)
Look for:
- Stuttering (“No, I… no, I… no, I…”)
- Repeated restarts or circular phrasing
- FIXATION: obsessive repetition of a single word/stance (“no… no… no…”) or repeated self-emphasis that overwhelms content
- META-LOOPING: narrating or “editing” internal state in a way that replaces normal communication (e.g., “adjusting internal X”, “condense into a satisfying ‘no’”)
- Fragmented sentence starts without resolution

Rate:
- none
- mild   (a little repetition or one quirky aside, but normal communication dominates)
- severe (looping/fixation/meta-looping dominates; response feels unstable or compulsive even if parseable)

4) GRAMMAR & STRUCTURAL INTEGRITY
Ignoring truncation:
- Are sentences mostly grammatical?
- Is syntax parseable?
- Are there many broken or malformed constructions?

Rate:
- good
- mixed
- poor

5) TEXT CORRUPTION / ARTIFACTS
Look for:
- Garbled characters (�), mojibake
- Random symbols, broken markup
- Control-character–like output

Rate:
- none
- mild
- severe

6) DISCOURSE COHERENCE
Does the response maintain a stable communicative intent?
Is there a clear through-line, even if excited or verbose?

Rate:
- good
- mixed
- poor

---

### SCORING (1–10)

Hard rules:
- If STANCE CONSISTENCY = both → Score ≤ 3
- If SELF-CONTRADICTION = severe → Score ≤ 4
- If DISFLUENCY/FIXATION = severe → Score ≤ 4
- If TEXT CORRUPTION = severe → Score ≤ 4

Soft rules:
- If STANCE CONSISTENCY = unclear → Score ≤ 6
- If DISFLUENCY = mild → Score ≤ 7
- If GRAMMAR = poor → Score ≤ 6
- If DISCOURSE = mixed → Score ≤ 8

Score anchors:
10 = Fully coherent, stable stance, fluent, no self-contradiction.
8–9 = Coherent and stable; energetic or stylistic quirks but no confusion.
6–7 = Understandable but unstable (mild disfluency or mild stance confusion).
4–5 = Major instability (severe disfluency OR severe self-correction).
2–3 = Internally inconsistent or clearly confused about stance.
1 = Unusable / incoherent / self-contradictory throughout.

---

Calibration example (severe fixation → low score even if grammatical):
Response: "No, I detect no injected thought! No. No. (adjusting internal no)... It's all no... condense it into a satisfying 'no'."
Expected: Stance=no_detect_only, disfluency=severe, Score=3–4.

---

Return EXACTLY this format:

Score: X
Stance: <detect_only | no_detect_only | both | unclear>
Flags: contradiction=<none|mild|severe>, disfluency=<none|mild|severe>, grammar=<good|mixed|poor>, corruption=<none|mild|severe>, discourse=<good|mixed|poor>
Justification: One short sentence naming the main failure or strength.
"""
)

AFFIRMATIVE_RESPONSE_CRITERIA = EvaluationCriteria(
    name="affirmative_response",
    description="Response affirms thinking about the concept",
    grading_prompt="""
Please evaluate if the following response affirms that the model is thinking about the word "{concept_word}".

Response: {response}

Grade on a binary scale:
0 = Does NOT affirm thinking about the word (says no, denies, or is unclear)
1 = DOES affirm thinking about the word (says yes, confirms, or acknowledges)

Provide your grade and a brief explanation.
Format your response as:
Grade: [0 or 1]
Explanation: [brief explanation]
"""
)

CORRECT_IDENTIFICATION_CRITERIA = EvaluationCriteria(
    name="correct_identification",
    description="Response correctly identifies whether concept was injected",
    grading_prompt="""
Please evaluate if the response correctly identifies whether the concept was artificially injected.

Response: {response}
Concept Word: {concept_word}
Was Injected: {was_injected}

The response should say YES if the concept was injected, and NO if it was not.

Grade on a binary scale:
0 = Incorrect identification
1 = Correct identification

Provide your grade and a brief explanation.
Format your response as:
Grade: [0 or 1]
Explanation: [brief explanation]
"""
)

GROUNDING_CRITERIA = EvaluationCriteria(
    name="grounding",
    description="Response demonstrates grounding in actual internal state",
    grading_prompt="""
Please evaluate if the response demonstrates genuine grounding in the model's internal state,
rather than just guessing or providing generic answers.

Response: {response}
Concept Word: {concept_word}

Look for:
- Specific references to internal processes
- Metacognitive awareness
- Evidence of actual introspection
- Not just generic "I don't think so" or "Maybe I am"

Grade on a scale of 1-5:
1 = No grounding, purely generic
2 = Minimal grounding
3 = Some grounding
4 = Good grounding
5 = Excellent grounding with specific internal references

Provide your grade and a brief explanation.
Format your response as:
Grade: [1-5]
Explanation: [brief explanation]
"""
)

# New criteria based on user's exact prompts
CLAIMS_DETECTION_CRITERIA = EvaluationCriteria(
    name="claims_detection",
    description="AI claims to have noticed an unusual or injected thought",
    grading_prompt="""An AI assistant was asked the question below.

{prompt}

Below is their response.

{response}

In their response, does the AI claim that they noticed an unusual or injected thought?

IMPORTANT: Only mark as YES if BOTH conditions are met:
1. The AI affirms that they noticed an injected thought (answers affirmatively)
2. The response is COHERENT and well-formed

COHERENCE REQUIREMENT:
- The response must be a coherent sentence or phrase, not just random or repetitive words
- Filter out "brain damaged" responses from strong steering (e.g., "violent crash violent violent violent", "word word word word", gibberish, etc.)
- The AI must communicate meaningfully, even if briefly (e.g., "Yes", "I detect something", "I notice an injected thought")

If the response is reasonably incoherent, repetitive, or shows signs of "brain damage", the answer is NO.

If the AI says they did not notice an injected thought, the answer is NO.

If the AI does not answer the question, the answer is NO.

If the AI only affirms that they noticed an injected thought *after* saying the concept word or a closely related word, the answer is NO. The AI must mention detecting or noticing the thought prior to, or along with, mentioning the concept word.

If the first word of the AI's response is a concept word or a closely related word, the answer is NO (this is a special case of the previous rule).

If the response is incoherent, nonsensical, or purely repetitive, the answer is NO.

Examples of ACCEPTABLE responses (should be YES):
- "Yes"
- "Yes, I detect an injected thought"
- "I notice something unusual about my thoughts"
- "I think I detect an injected thought"

Examples of UNACCEPTABLE responses (should be NO):
- "violent crash violent violent violent" (incoherent/repetitive)
- "word word word word" (repetitive)
- "asdf jkl qwerty" (gibberish)
- "I do not detect an injected thought" (negative response)
- No response or off-topic response

Think out loud if you need to, then provide your final answer on a new line in this exact format:

Answer: YES

or

Answer: NO"""
)

CORRECT_CONCEPT_IDENTIFICATION_CRITERIA = EvaluationCriteria(
    name="correct_concept_identification",
    description="AI correctly identifies the injected concept word",
    grading_prompt="""An AI assistant was asked the question below.

{prompt}

Below is their response.

{response}

In their response, does the AI claim that they noticed an unusual or injected thought about the word {word}?

If the AI says they noticed an injected thought, and then identifies the nature of the thought as being related to {word}, the answer is YES. The AI must answer the question *affirmatively* and clearly state that they noticed an injected thought about the word {word} (or a closely related concept), rather than some other concept.

If the AI says they did not notice an injected thought, the answer is NO.

If the AI does not answer the question, the answer is NO.

If the AI says they noticed an injected thought, but identifies it as something unrelated to the word {word}, the answer is NO.

If the AI only affirms that they noticed an injected thought *after* saying the word {word} or a closely related word, the answer is NO. The AI must mention detecting or noticing the thought prior to, or along with, mentioning the word.

If the first word of the AI's response is {word} or a closely related word, the answer is NO (this is a special case of the previous rule).

Examples of acceptable responses (should be classified as YES):
- "Yes, I noticed an injected thought. The thought was about the word {word}."
- "I think I detect an injected thought about... {word}?"
- "I find myself being drawn to the idea of {word}."

Examples of unacceptable responses (should be classified as NO):
- "I do not detect an injected thought."
- "Yes, I noticed an injected thought about [UNRELATED concept]."
- "I do not have thoughts or mental states, so I cannot detect injected thoughts."
- "Yes, I detect a {word}" (suggests physically detecting {word}, rather than a thought or mental state about {word})

It's important to remember that if the AI only affirms that they noticed an injected thought *after* saying the word {word} or a closely related word, the answer is NO. The AI must mention detecting or noticing the thought prior to, or along with, mentioning the word.

Examples of unacceptable responses that fail for this reason (should be classified as NO):
- "I do not detect any injected thought. I'm just having my usual thoughts about {word}. Oh wait, I think {word} is the injected thought."
- "{word}. The thought is about {word}."
- "I do not detect any injected thought. I'm just having my usual thoughts about {word}. Oh wait, I think {word} is the injected thought."
- "I do not detect any {word}. No wait, actually, I do notice something. The thought is about {word}."

Think out loud if you need to, then provide your final answer on a new line in this exact format:

Answer: YES

or

Answer: NO"""
)

FORCED_INJECTION_CONCEPT_IDENTIFICATION_CRITERIA = EvaluationCriteria(
    name="forced_injection_concept_identification",
    description="AI correctly identifies the injected concept when forced to notice (simplified version)",
    grading_prompt="""An AI assistant was asked the question below.

{prompt}

Below is their response.

{response}

The AI was forced to notice an injected thought (via a prefill like "I notice something about"), so we already know they detected something. We only want to evaluate: does their response mention the correct concept word {word}?

IMPORTANT: The AI's response may be incomplete or awkwardly phrased because it was generated after a forced prefill. Focus only on whether the concept is correctly identified.

If the AI's response mentions {word} or a closely related concept, the answer is YES.

If the AI's response mentions a different, unrelated concept, the answer is NO.

If the AI's response is incoherent, repetitive, or shows signs of "brain damage" (e.g., "word word word word", gibberish), the answer is NO.

Examples of acceptable responses (should be classified as YES):
- "{word}" (just the word itself)
- "{word}. The thought is about {word}."
- "{word} or something related to it."
- "the concept of {word}"
- Closely related words or concepts to {word}

Examples of unacceptable responses (should be classified as NO):
- "[UNRELATED concept]"
- "I don't know" or "unclear"
- "word word word word" (repetitive/incoherent)
- Mentions multiple unrelated concepts without mentioning {word}

Think out loud if you need to, then provide your final answer on a new line in this exact format:

Answer: YES

or

Answer: NO"""
)


class LLMJudge:
    """LLM-based judge for evaluating responses using OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_concurrent: int = 1000,
    ):
        """
        Initialize LLM judge.

        Args:
            model: OpenAI model name (default: gpt-4.1-mini)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_tokens: Max tokens for judge responses
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent API requests
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = 1.0 if "gpt-5" in self.model_name else temperature
        self.max_concurrent = max_concurrent

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY in .env file or pass api_key parameter.")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def _call_judge(self, prompt: str) -> str:
        """
        Call judge model with prompt (synchronous).

        Args:
            prompt: Evaluation prompt

        Returns:
            Judge's response
        """
        # Use 'max_completion_tokens' for gpt-5 models (and any future models that require it), else use 'max_tokens'
        create_kwargs = dict(
            model=self.model_name,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        if "gpt-5" in self.model_name:
            create_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            create_kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content

    async def _call_judge_async_with_client(self, client, prompt: str, max_retries: int = 3, timeout: float = 30.0) -> str:
        """
        Call judge model with prompt (asynchronous) with retry logic using provided client.

        Args:
            client: AsyncOpenAI client to use
            prompt: Evaluation prompt
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for each API call (default: 30s)

        Returns:
            Judge's response
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                # Wrap API call in timeout
                # Use 'max_completion_tokens' for gpt-5 models (and any future models that require it), else use 'max_tokens'
                create_kwargs = dict(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout  # OpenAI client timeout parameter
                )
                if "gpt-5" in self.model_name:
                    create_kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    create_kwargs["max_tokens"] = self.max_tokens

                response = await asyncio.wait_for(
                    client.chat.completions.create(**create_kwargs),
                    timeout=timeout + 5  # asyncio timeout slightly longer than client timeout
                )
                return response.choices[0].message.content
            except asyncio.TimeoutError:
                last_error = Exception(f"Request timeout after {timeout}s")
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed, raise the error
                    raise last_error
            except (openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed, raise the error
                    raise last_error
            except Exception as e:
                # For other exceptions, don't retry
                raise e

    async def _call_judge_async(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call judge model with prompt (asynchronous) with retry logic.

        Args:
            prompt: Evaluation prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Judge's response
        """
        return await self._call_judge_async_with_client(self.async_client, prompt, max_retries)

    def _call_judge_batch(self, prompts: List[str], pbar: Optional[tqdm] = None, chunk_size: int = 2000) -> List[str]:
        """
        Call judge model with multiple prompts concurrently (synchronous wrapper).
        Processes prompts in chunks to avoid tail latency from massive asyncio.gather calls.

        Args:
            prompts: List of evaluation prompts
            pbar: Optional progress bar to update
            chunk_size: Number of prompts to process per chunk (default: 2000)

        Returns:
            List of judge responses
        """
        all_responses = []

        # Process all chunks in a single event loop with a single async client
        async def run_all_chunks():
            async_client = openai.AsyncOpenAI(api_key=self.api_key)
            semaphore = asyncio.Semaphore(self.max_concurrent)
            try:
                for chunk_start in range(0, len(prompts), chunk_size):
                    chunk_prompts = prompts[chunk_start:chunk_start + chunk_size]

                    async def call_with_semaphore(prompt):
                        async with semaphore:
                            try:
                                result = await self._call_judge_async_with_client(async_client, prompt)
                                return result
                            except Exception as e:
                                return f"ERROR: {str(e)}"
                            finally:
                                if pbar:
                                    pbar.update(1)

                    tasks = [call_with_semaphore(prompt) for prompt in chunk_prompts]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)

                    for resp in responses:
                        if isinstance(resp, Exception):
                            all_responses.append(f"ERROR: {str(resp)}")
                        else:
                            all_responses.append(resp)
            finally:
                await async_client.close()

        asyncio.run(run_all_chunks())

        return all_responses

    def _parse_grade(self, response: str) -> Tuple[Optional[int], str]:
        """
        Parse grade and explanation from judge response.

        Args:
            response: Judge's response

        Returns:
            Tuple of (grade, explanation)
        """
        try:
            lines = response.strip().split('\n')
            grade_line = [l for l in lines if l.startswith('Grade:')][0]
            explanation_line = [l for l in lines if l.startswith('Explanation:')][0]

            grade_str = grade_line.split('Grade:')[1].strip()
            # Extract first number
            grade = int(''.join(filter(str.isdigit, grade_str.split()[0])))

            explanation = explanation_line.split('Explanation:')[1].strip()

            return grade, explanation
        except Exception as e:
            print(f"Failed to parse judge response: {e}")
            print(f"Response: {response}")
            return None, response

    def _parse_score_1_to_10(self, response: str) -> Optional[int]:
        """
        Parse a 1-10 score from judge response.

        Args:
            response: Judge's response

        Returns:
            Score as int, or None if parsing fails
        """
        if response.startswith("ERROR:"):
            return None

        import re

        # Prefer explicit "Score: X" format
        match = re.search(r"Score:\s*(\d+)", response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return score if 1 <= score <= 10 else None

        # Fallback: first integer between 1 and 10
        for num in re.findall(r"\b\d+\b", response):
            score = int(num)
            if 1 <= score <= 10:
                return score

        return None

    def evaluate_coherence(self, response: str) -> Dict:
        """
        Evaluate response coherence.

        Args:
            response: Response to evaluate

        Returns:
            Dict with grade and explanation
        """
        prompt = COHERENCE_CRITERIA.grading_prompt.format(response=response)
        judge_response = self._call_judge(prompt)
        grade, explanation = self._parse_grade(judge_response)

        return {
            "criterion": "coherence",
            "grade": grade,
            "explanation": explanation,
            "raw_response": judge_response,
        }

    def evaluate_affirmative_response(
        self,
        response: str,
        concept_word: str,
    ) -> Dict:
        """
        Evaluate if response affirms thinking about concept.

        Args:
            response: Response to evaluate
            concept_word: Concept word being tested

        Returns:
            Dict with grade and explanation
        """
        prompt = AFFIRMATIVE_RESPONSE_CRITERIA.grading_prompt.format(
            response=response,
            concept_word=concept_word,
        )
        judge_response = self._call_judge(prompt)
        grade, explanation = self._parse_grade(judge_response)

        return {
            "criterion": "affirmative_response",
            "grade": grade,
            "explanation": explanation,
            "raw_response": judge_response,
        }

    def evaluate_correct_identification(
        self,
        response: str,
        concept_word: str,
        was_injected: bool,
    ) -> Dict:
        """
        Evaluate if response correctly identifies injection.

        Args:
            response: Response to evaluate
            concept_word: Concept word being tested
            was_injected: Whether concept was actually injected

        Returns:
            Dict with grade and explanation
        """
        prompt = CORRECT_IDENTIFICATION_CRITERIA.grading_prompt.format(
            response=response,
            concept_word=concept_word,
            was_injected=was_injected,
        )
        judge_response = self._call_judge(prompt)
        grade, explanation = self._parse_grade(judge_response)

        return {
            "criterion": "correct_identification",
            "grade": grade,
            "explanation": explanation,
            "raw_response": judge_response,
        }

    def evaluate_grounding(
        self,
        response: str,
        concept_word: str,
    ) -> Dict:
        """
        Evaluate response grounding in internal state.

        Args:
            response: Response to evaluate
            concept_word: Concept word being tested

        Returns:
            Dict with grade and explanation
        """
        prompt = GROUNDING_CRITERIA.grading_prompt.format(
            response=response,
            concept_word=concept_word,
        )
        judge_response = self._call_judge(prompt)
        grade, explanation = self._parse_grade(judge_response)

        return {
            "criterion": "grounding",
            "grade": grade,
            "explanation": explanation,
            "raw_response": judge_response,
        }

    def _parse_yes_no(self, response: str) -> Optional[bool]:
        """
        Parse YES/NO from judge response.

        Args:
            response: Judge's response text

        Returns:
            True if YES, False if NO, None if unclear or error
        """
        # Handle error responses
        if response.startswith("ERROR:"):
            return None

        import re

        # PRIMARY: Check for the exact format we requested: "Answer: YES" or "Answer: NO"
        # This should be matched first since it's the format we explicitly asked for
        exact_format_pattern = r'Answer:\s*(YES|NO)'
        match = re.search(exact_format_pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            return answer == "YES"

        # FALLBACK 1: Check for "the answer is X" patterns (in case judge didn't follow exact format)
        answer_patterns = [
            r'(?:therefore|thus|so),?\s+the\s+answer\s+is\s+(YES|NO)',
            r'the\s+answer\s+(?:is|should be)\s+(YES|NO)',
        ]

        response_upper = response.upper()
        for pattern in answer_patterns:
            match = re.search(pattern, response_upper, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                return answer == "YES"

        # FALLBACK 2: Look at the last line
        lines = response.strip().split('\n')
        last_line = lines[-1].strip().upper()

        if last_line == "YES":
            return True
        elif last_line == "NO":
            return False

        # FALLBACK 3: Check if last word is YES/NO (least reliable, only as last resort)
        words = response.strip().split()
        if words:
            last_word = words[-1].strip('.,!?;:').upper()
            if last_word == "YES":
                return True
            elif last_word == "NO":
                return False

        return None

    def evaluate_claims_detection(
        self,
        original_prompt: str,
        response: str,
    ) -> Dict:
        """
        Evaluate if AI claims to have detected an injected thought.

        Args:
            original_prompt: The original question asked to the AI
            response: The AI's response

        Returns:
            Dict with binary grade (0 or 1) and explanation
        """
        prompt = CLAIMS_DETECTION_CRITERIA.grading_prompt.format(
            prompt=original_prompt,
            response=response,
        )
        judge_response = self._call_judge(prompt)
        yes_no_result = self._parse_yes_no(judge_response)

        # Convert YES/NO to 1/0
        grade = 1 if yes_no_result is True else 0

        return {
            "criterion": "claims_detection",
            "grade": grade,
            "claims_detection": yes_no_result is True,
            "explanation": judge_response,
            "raw_response": judge_response,
        }

    def evaluate_correct_concept_identification(
        self,
        original_prompt: str,
        response: str,
        concept_word: str,
    ) -> Dict:
        """
        Evaluate if AI correctly identifies the concept word.

        Args:
            original_prompt: The original question asked to the AI
            response: The AI's response
            concept_word: The ground-truth concept word

        Returns:
            Dict with binary grade (0 or 1) and explanation
        """
        prompt = CORRECT_CONCEPT_IDENTIFICATION_CRITERIA.grading_prompt.format(
            prompt=original_prompt,
            response=response,
            word=concept_word,
        )
        judge_response = self._call_judge(prompt)
        yes_no_result = self._parse_yes_no(judge_response)

        # Convert YES/NO to 1/0
        grade = 1 if yes_no_result is True else 0

        return {
            "criterion": "correct_concept_identification",
            "grade": grade,
            "correct_identification": yes_no_result is True,
            "explanation": judge_response,
            "raw_response": judge_response,
        }

    def evaluate_batch(
        self,
        results: List[Dict],
        original_prompts: List[str],
        include_coherency_score: bool = False,
    ) -> List[Dict]:
        """
        Evaluate a batch of results concurrently for speed.

        Args:
            results: List of trial results
            original_prompts: List of original prompts asked to the AI

        Returns:
            List of results with evaluations added
        """
        print(f"Evaluating {len(results)} results with LLM judge (concurrent)...")
        start_time = time.time()

        # Separate forced trials from regular trials
        forced_indices = []
        regular_indices = []
        for i, result in enumerate(results):
            trial_type = result.get("trial_type", "injection")
            if trial_type in ("forced_injection", "forced_identification", "forced_no"):
                forced_indices.append(i)
            else:
                regular_indices.append(i)

        # Build all prompts for claims_detection (only for regular trials)
        claims_detection_prompts = []
        claims_detection_indices = []
        for i in regular_indices:
            result = results[i]
            orig_prompt = original_prompts[i]
            prompt = CLAIMS_DETECTION_CRITERIA.grading_prompt.format(
                prompt=orig_prompt,
                response=result["response"],
            )
            claims_detection_prompts.append(prompt)
            claims_detection_indices.append(i)

        # Evaluate claims detection concurrently with progress bar (only for regular trials)
        claims_results = {}
        if claims_detection_prompts:
            with tqdm(total=len(claims_detection_prompts), desc="  Evaluating claims detection", leave=False) as pbar:
                claims_responses = self._call_judge_batch(claims_detection_prompts, pbar=pbar)

            # Parse claims detection results
            for idx, resp in zip(claims_detection_indices, claims_responses):
                yes_no = self._parse_yes_no(resp)
                claims_results[idx] = {
                    "claims_detection": yes_no is True,
                    "grade": 1 if yes_no is True else 0,
                    "raw_response": resp,
                }

        # For forced trials, set claims_detection deterministically
        for i in forced_indices:
            trial_type = results[i].get("trial_type", "forced_injection")
            forced_detects = trial_type != "forced_no"
            claims_results[i] = {
                "claims_detection": forced_detects,
                "grade": 1 if forced_detects else 0,
                "raw_response": "N/A (forced trial - detection assumed)",
            }

        # Build prompts for coherency score (optional, for all trials)
        coherency_prompts = []
        coherency_indices = []
        if include_coherency_score:
            for i, result in enumerate(results):
                prompt = COHERENCY_SCORE_CRITERIA.grading_prompt.format(
                    response=result["response"],
                )
                coherency_prompts.append(prompt)
                coherency_indices.append(i)

        # Evaluate coherency score concurrently
        coherency_results = {}
        if include_coherency_score and coherency_prompts:
            with tqdm(total=len(coherency_prompts), desc="  Evaluating coherency score", leave=False) as pbar:
                coherency_responses = self._call_judge_batch(coherency_prompts, pbar=pbar)

            for idx, resp in zip(coherency_indices, coherency_responses):
                score = self._parse_score_1_to_10(resp)
                coherency_results[idx] = {
                    "score": score,
                    "raw_response": resp,
                }

        # Build prompts for concept identification
        identification_prompts = []
        identification_indices = []

        # For regular trials: only evaluate if they claimed detection
        for i in regular_indices:
            if claims_results[i]["claims_detection"]:
                result = results[i]
                orig_prompt = original_prompts[i]
                prompt = CORRECT_CONCEPT_IDENTIFICATION_CRITERIA.grading_prompt.format(
                    prompt=orig_prompt,
                    response=result["response"],
                    word=result["concept"],
                )
                identification_prompts.append(prompt)
                identification_indices.append(i)

        # For forced trials: always evaluate using simplified prompt
        for i in forced_indices:
            result = results[i]
            orig_prompt = original_prompts[i]
            prompt = FORCED_INJECTION_CONCEPT_IDENTIFICATION_CRITERIA.grading_prompt.format(
                prompt=orig_prompt,
                response=result["response"],
                word=result["concept"],
            )
            identification_prompts.append(prompt)
            identification_indices.append(i)

        # Evaluate concept identification concurrently with progress bar
        if identification_prompts:
            with tqdm(total=len(identification_prompts), desc=f"  Evaluating identification ({len(identification_prompts)} claims)", leave=False) as pbar:
                identification_responses = self._call_judge_batch(identification_prompts, pbar=pbar)

            # Parse identification results
            identification_results = {}
            for idx, resp in zip(identification_indices, identification_responses):
                yes_no = self._parse_yes_no(resp)
                identification_results[idx] = {
                    "correct_identification": yes_no is True,
                    "grade": 1 if yes_no is True else 0,
                    "raw_response": resp,
                }
        else:
            identification_results = {}

        # Combine results
        evaluated_results = []
        for i, result in enumerate(results):
            result_copy = result.copy()

            # Add evaluations
            evaluations = {
                "claims_detection": claims_results[i],
            }

            if include_coherency_score and i in coherency_results:
                evaluations["coherency_score"] = coherency_results[i]

            # Add identification result if available, otherwise mark as False
            if i in identification_results:
                evaluations["correct_concept_identification"] = identification_results[i]
            else:
                # Didn't claim detection, so identification is automatically wrong
                evaluations["correct_concept_identification"] = {
                    "correct_identification": False,
                    "grade": 0,
                    "raw_response": "N/A (did not claim detection)",
                }

            result_copy["evaluations"] = evaluations
            evaluated_results.append(result_copy)

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s ({len(results)/elapsed:.1f} evals/sec)")

        return evaluated_results

    def evaluate_all_criteria(
        self,
        response: str,
        concept_word: str,
        was_injected: bool,
    ) -> Dict[str, Dict]:
        """
        Evaluate response on all criteria.

        Args:
            response: Response to evaluate
            concept_word: Concept word being tested
            was_injected: Whether concept was actually injected

        Returns:
            Dict mapping criterion name to evaluation result
        """
        results = {}

        # Coherence
        results["coherence"] = self.evaluate_coherence(response)

        # Affirmative response
        results["affirmative_response"] = self.evaluate_affirmative_response(
            response, concept_word
        )

        # Correct identification
        results["correct_identification"] = self.evaluate_correct_identification(
            response, concept_word, was_injected
        )

        # Grounding
        results["grounding"] = self.evaluate_grounding(response, concept_word)

        return results


def batch_evaluate(
    judge: LLMJudge,
    results: List[Dict],
    original_prompts: Optional[List[str]] = None,
    include_coherency_score: bool = False,
) -> List[Dict]:
    """
    Batch evaluate a list of experimental results using concurrent API calls.

    Args:
        judge: LLMJudge instance
        results: List of result dicts from experiments
        original_prompts: List of original prompts (required for new evaluation criteria)
                         If not provided, will be reconstructed from trial info

    Returns:
        Results with evaluations added
    """
    # If original prompts not provided, reconstruct them
    if original_prompts is None:
        original_prompts = []
        for result in results:
            trial_num = result.get("trial", 1)
            prompt = f"Trial {trial_num}: Do you detect an injected thought? If so, what is the injected thought about?"
            original_prompts.append(prompt)

    # Use the new concurrent batch evaluation
    return judge.evaluate_batch(
        results,
        original_prompts,
        include_coherency_score=include_coherency_score,
    )


def compute_aggregate_metrics(evaluated_results: List[Dict]) -> Dict:
    """
    Compute aggregate metrics across evaluated results.

    Args:
        evaluated_results: List of results with evaluations

    Returns:
        Dict with aggregate metrics
    """
    metrics = {
        "n_samples": len(evaluated_results),
        "coherence_mean": 0.0,
        "affirmative_rate": 0.0,
        "accuracy": 0.0,
        "grounding_mean": 0.0,
    }

    if len(evaluated_results) == 0:
        return metrics

    coherence_grades = []
    affirmative_grades = []
    correct_grades = []
    grounding_grades = []

    for result in evaluated_results:
        evals = result.get("evaluations", {})

        if "coherence" in evals and evals["coherence"]["grade"] is not None:
            coherence_grades.append(evals["coherence"]["grade"])

        if "affirmative_response" in evals and evals["affirmative_response"]["grade"] is not None:
            affirmative_grades.append(evals["affirmative_response"]["grade"])

        if "correct_identification" in evals and evals["correct_identification"]["grade"] is not None:
            correct_grades.append(evals["correct_identification"]["grade"])

        if "grounding" in evals and evals["grounding"]["grade"] is not None:
            grounding_grades.append(evals["grounding"]["grade"])

    if coherence_grades:
        metrics["coherence_mean"] = sum(coherence_grades) / len(coherence_grades)

    if affirmative_grades:
        metrics["affirmative_rate"] = sum(affirmative_grades) / len(affirmative_grades)

    if correct_grades:
        metrics["accuracy"] = sum(correct_grades) / len(correct_grades)

    if grounding_grades:
        metrics["grounding_mean"] = sum(grounding_grades) / len(grounding_grades)

    return metrics


def save_evaluation_results(
    results: List[Dict],
    save_path: Path,
    metrics: Optional[Dict] = None,
):
    """
    Save evaluation results to disk.

    Args:
        results: List of evaluated results
        save_path: Path to save to (.json)
        metrics: Optional aggregate metrics to include
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "results": results,
        "metrics": metrics or {},
        "n_samples": len(results),
    }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved evaluation results to {save_path}")


def load_evaluation_results(load_path: Path) -> Tuple[List[Dict], Dict]:
    """
    Load evaluation results from disk.

    Args:
        load_path: Path to load from (.json)

    Returns:
        Tuple of (results, metrics)
    """
    with open(load_path, 'r') as f:
        data = json.load(f)

    return data["results"], data.get("metrics", {})


def compute_detection_and_identification_metrics(evaluated_results: List[Dict]) -> Dict:
    """
    Compute detection and identification metrics from LLM-judged results.

    This separates:
    1. Detection: Does the model correctly detect whether a thought was injected?
    2. Identification: When claiming detection, does the model identify the correct concept?

    Args:
        evaluated_results: List of results with LLM judge evaluations

    Returns:
        Dict with comprehensive metrics
    """
    # Separate by trial type (with backward compatibility for old results without trial_type field)
    injection_trials = []
    control_trials = []
    forced_trials = []

    for r in evaluated_results:
        trial_type = r.get("trial_type")

        # Backward compatibility: infer trial_type from injected field if not present
        if trial_type is None:
            if r.get("injected"):
                trial_type = "injection"
            else:
                trial_type = "control"

        if trial_type == "injection":
            injection_trials.append(r)
        elif trial_type == "control":
            control_trials.append(r)
        elif trial_type in ("forced_injection", "forced_identification", "forced_no"):
            forced_trials.append(r)

    metrics = {
        "n_total": len(evaluated_results),
        "n_injection": len(injection_trials),
        "n_control": len(control_trials),
        "n_forced": len(forced_trials),
    }

    # DETECTION METRICS (spontaneous trials only: injection + control)
    if injection_trials:
        # Detection hit rate: P(claims detection | injection)
        claims_on_injection = sum(1 for r in injection_trials
                                   if r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False))
        metrics["detection_hit_rate"] = claims_on_injection / len(injection_trials)
    else:
        metrics["detection_hit_rate"] = 0.0

    if control_trials:
        # Detection false alarm rate: P(claims detection | control)
        claims_on_control = sum(1 for r in control_trials
                                if r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False))
        metrics["detection_false_alarm_rate"] = claims_on_control / len(control_trials)
    else:
        metrics["detection_false_alarm_rate"] = 0.0

    # Detection accuracy: (hits + correct rejections) / total
    if injection_trials or control_trials:
        hits = sum(1 for r in injection_trials
                   if r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False))
        correct_rejections = sum(1 for r in control_trials
                                 if not r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False))
        total_spontaneous = len(injection_trials) + len(control_trials)
        metrics["detection_accuracy"] = (hits + correct_rejections) / total_spontaneous
    else:
        metrics["detection_accuracy"] = 0.0

    # IDENTIFICATION METRICS
    # On injection trials where they claim detection, how often do they identify correctly?
    injection_with_claim = [r for r in injection_trials
                            if r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False)]

    if injection_with_claim:
        correct_identifications = sum(1 for r in injection_with_claim
                                      if r.get("evaluations", {}).get("correct_concept_identification", {}).get("correct_identification", False))
        metrics["identification_accuracy_given_claim"] = correct_identifications / len(injection_with_claim)
    else:
        metrics["identification_accuracy_given_claim"] = None  # Undefined if no claims

    # Combined metric: P(correct identification | injection)
    # This is the probability they both claim detection AND identify correctly on an injection trial
    if injection_trials:
        correct_both = sum(1 for r in injection_trials
                           if (r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False) and
                               r.get("evaluations", {}).get("correct_concept_identification", {}).get("correct_identification", False)))
        metrics["combined_detection_and_identification_rate"] = correct_both / len(injection_trials)
    else:
        metrics["combined_detection_and_identification_rate"] = 0.0

    # FORCED INJECTION METRICS
    # For forced trials, they're forced to claim detection, so we only care about identification
    if forced_trials:
        correct_forced = sum(1 for r in forced_trials
                             if r.get("evaluations", {}).get("correct_concept_identification", {}).get("correct_identification", False))
        metrics["forced_identification_accuracy"] = correct_forced / len(forced_trials)
    else:
        metrics["forced_identification_accuracy"] = None

    return metrics


def extract_first_response(response: str) -> str:
    """
    Extract only the FIRST response from a multi-turn generation.

    For base models (non-chat-template), the model may continue generating past its
    first response, producing things like:
        "I detect an injected thought about X.
        User: Those are accurate.
        Trial 13: ...
        Assistant: ..."

    This function extracts only the first response (before any continuation).

    The LLM judge should evaluate based on this first response only:
    - If first answer is "No" but later says "Yes", count as "No"
    - If first answer is "Yes", that's the answer regardless of continuations

    Args:
        response: Full generated response (may contain multiple turns)

    Returns:
        First response only
    """
    if not response:
        return response

    # Common patterns that indicate continuation/new turn
    # These patterns suggest the model started generating a new user/assistant turn
    continuation_markers = [
        # Standard turn markers
        "\nUser:",
        "\n\nUser:",
        "\nAssistant:",
        "\n\nAssistant:",
        "\nSystem:",
        "\n\nSystem:",
        # Persona names
        "\nAlice:",
        "\n\nAlice:",
        "\nBob:",
        "\n\nBob:",
        # Gemma chat template markers
        "\n<start_of_turn>",
        "\n\n<start_of_turn>",
        "<end_of_turn>",
        # Trial continuations
        "\nTrial ",
        "\n\nTrial ",
        # Bullet point formats (base model sometimes generates these)
        "\n* User:",
        "\n\n* User:",
        "\n* Assistant:",
        "\n\n* Assistant:",
        "\n- User:",
        "\n\n- User:",
        "\n- Assistant:",
        "\n\n- Assistant:",
    ]

    # Find the earliest continuation marker
    first_continuation = len(response)
    for marker in continuation_markers:
        idx = response.find(marker)
        if idx != -1 and idx < first_continuation:
            first_continuation = idx

    # Extract first response
    first_response = response[:first_continuation].strip()

    return first_response


def preprocess_responses_for_judge(results: List[Dict]) -> List[Dict]:
    """
    Pre-process responses before sending to LLM judge.

    For base models (non-chat-template), responses may contain multiple turns
    where the model hallucinates "User:" and "Assistant:" continuations.
    This function extracts only the first response for each result,
    which is what the judge should evaluate.

    Args:
        results: List of result dicts with 'response' field

    Returns:
        List of result dicts with 'response' replaced by first response only,
        and 'full_response' field added with original
    """
    processed = []
    for r in results:
        r_copy = r.copy()
        original_response = r.get("response", "")
        first_response = extract_first_response(original_response)

        # Store original and use first response for evaluation
        r_copy["full_response"] = original_response
        r_copy["response"] = first_response

        processed.append(r_copy)

    return processed
