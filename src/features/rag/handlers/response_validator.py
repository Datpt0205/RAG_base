import time
import json
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.utils.logger.custom_logging import LoggerMixin


class ValidationResult(str, Enum):
    VALID = "VALID"             
    NEEDS_REFINEMENT = "NEEDS_REFINEMENT"  
    INVALID = "INVALID"


class ResponseValidator(LoggerMixin):
    def __init__(self):
        super().__init__()
    
    async def validate_response(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        llm = None,
        original_question: Optional[str] = None
    ) -> Tuple[ValidationResult, Optional[str], Optional[Dict[str, Any]]]:
        """
        Evaluate the quality of the answer and suggest improvements if necessary

        Args:
            query: Original query
            response: Response from LLM
            context: Relevant context (optional)
            llm: LLM model to perform evaluation

        Returns:
            Tuple[ValidationResult, Optional[str], Optional[Dict]]:
            - Evaluation result
            - Suggested query rewrite (if necessary)
            - Evaluation details
        """
        if not llm:
            self.logger.error("LLM is not provided to the validator.")
            return ValidationResult.VALID, None, {"error": "LLM not offered"}
            
        self.logger.info(f"Start evaluating the answer to the query: '{query}'")
        start_time = time.time()
        
        original_context = ""
        if original_question and original_question != query:
            original_context = f"Original user question: {original_question}\nRewritten query for retrieval: {query}"
        else:
            original_context = f"Query: {query}"
            
        system_prompt = """You are an expert answer validator with a critical eye for detail and accuracy. Your task is to assess whether the provided answer fully and correctly addresses the original query based on the available context.

Please evaluate the answer using these criteria:
1. Relevance: Does the answer directly address the query?
2. Completeness: Does the answer provide all the information requested?
3. Accuracy: Is the answer factually correct based on the provided context?
4. Clarity: Is the answer clear and well-articulated?

Return your evaluation as a JSON object with the following structure:
{
  "validation_result": "VALID" | "NEEDS_REFINEMENT" | "INVALID",
  "reasoning": "Detailed explanation of your assessment",
  "missing_information": "What information is missing (if any)",
  "improved_query_suggestion": "A suggested improvement to the original query to get better results (if needed)"
}

Where:
- VALID: The answer is complete, accurate and directly addresses the query
- NEEDS_REFINEMENT: The answer partially addresses the query but needs improvement
- INVALID: The answer is incorrect, irrelevant, or completely fails to address the query

Your job is critical as it determines whether to accept the current answer or try to improve it.
"""
        
        context_section = ""
        if context:
            context_section = f"\n\nAvailable Context:\n{context}"
            
        human_prompt = f"""{original_context}

Response to evaluate: {response}{context_section}

PLEASE EVALUATE the ANSWER and provide your assessment in the JSON FORMAT described."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            raw_assessment = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            assessment = self._extract_json(raw_assessment)
            
            if not assessment or "validation_result" not in assessment:
                self.logger.warning("The evaluation results cannot be analyzed and are considered valid.")
                assessment = {
                    "validation_result": "VALID",
                    "reasoning": "Cannot parse validator response, defaulting to valid."
                }
                
            # Normalize validation_result
            validation_result = assessment.get("validation_result", "VALID").upper()
            if validation_result not in [vr.value for vr in ValidationResult]:
                self.logger.warning(f"Invalid validation_result value: {validation_result}")
                validation_result = "VALID"
                
            # Convert to enum
            result = ValidationResult(validation_result)
            
            # Prepare new query hints if needed
            improved_query = assessment.get("improved_query_suggestion", None)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Evaluation completed in {elapsed_time:.3f}s - Result: {result.value}")
            
            if result != ValidationResult.VALID:
                self.logger.info(f"Suggested query improvements: {improved_query}")
                
            return result, improved_query, assessment
            
        except Exception as e:
            self.logger.error(f"Error in evaluating answer: {str(e)}")
            return ValidationResult.VALID, None, {"error": str(e)}
            
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text (can return markdown)

        Args:
            text: Text to extract

        Returns:
            Optional[Dict[str, Any]]: JSON object or None if unable to extract
        """
        try:
            return json.loads(text)
        except:
            pass
            
        try:
            # Search JSON syntax in markdown
            import re
            json_pattern = r"```(?:json)?(.*?)```"
            match = re.search(json_pattern, text, re.DOTALL)
            
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
                
            # Search by curly brackets
            json_pattern = r"\{.*\}"
            match = re.search(json_pattern, text, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
                
        except Exception as e:
            self.logger.error(f"Error while extracting JSON: {str(e)}")
            
        return None
        
    async def suggest_improved_query(
        self,
        original_query: str,
        original_response: str,
        missing_info: str,
        llm = None
    ) -> str:
        """
        Suggested query improvements based on missing information

        Args:
            original_query: Original query
            original_response: Current response
            missing_info: Missing information
            llm: LLM model for suggestion

        Returns:
            str: Improved query
        """
        if not llm:
            self.logger.warning("LLM is not provided for query improvement suggestions.")
            return original_query
            
        try:
            system_prompt = """You are a search expert who helps improve queries to retrieve better information. Your job is to make a query more specific and targeted to find missing information.

When improving a query:
1. Focus on the exact information that is missing from the previous answer
2. Make the query clear and specific
3. Keep the query concise and to the point
4. Include important context from the original query
5. Use search-friendly phrasing

Return ONLY the improved query text with no explanations or additional information.
"""
            
            human_prompt = f"""Original Query: {original_query}

Previous Answer: {original_response}

Missing Information: {missing_info}

Please provide an improved query that will help retrieve the missing information."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            improved_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            if not improved_query or len(improved_query) < 5:
                self.logger.warning("Suggestions to improve invalid query, keep original query")
                return original_query
                
            self.logger.info(f"Original Query: '{original_query}' -> Improvement: '{improved_query}'")
            return improved_query
            
        except Exception as e:
            self.logger.error(f"Error when suggesting query improvement: {str(e)}")
            return original_query