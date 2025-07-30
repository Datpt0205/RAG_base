import re
import asyncio
from typing import AsyncGenerator
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from src.core.utils.logger.custom_logging import LoggerMixin
from src.features.rag.helpers.llm_helper import LLMGenerator
from src.features.rag.handlers.retrieval_handler import default_search_retrieval

class ActionType(Enum):
    """Enum for different types of actions the agent can take"""
    RETRIEVAL = "retrieval"  # RAG action
    THINK = "think"
    GENERATE_RESPONSE = "generate_response"

class ReActAgent(LoggerMixin):
    """
    Implementation of ReAct (Reasoning and Acting) pattern.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_generator = LLMGenerator()
        
    async def plan(self, query: str, context: Optional[str] = None, 
                  model_name: str = "llama3.1:8b") -> Dict[str, Any]:
        """Generate a plan based on the query and available context"""
        self.logger.info(f"=== PLANNING PHASE ===")

        llm = await self.llm_generator.get_llm(model=model_name)
        
        planning_prompt = self._create_planning_prompt(query, context)
        response = await llm.ainvoke(planning_prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        self.logger.info(f"=== PLANNING RESULT ===\n{content}")

        actions = self._parse_actions(content)
        self.logger.info(f"Parsed {len(actions)} actions from plan")
        for i, action in enumerate(actions):
            self.logger.info(f"Action {i+1}: {action['type']} - Query: {action['query']}")
        
        return {
            "query": query,
            "plan": content,
            "actions": actions
        }
    
    async def execute_plan(self, 
                           plan: Dict[str, Any], 
                           collection_name: str,
                           context: Optional[str] = None,
                           model_name: str = "llama3.1:8b") -> Dict[str, Any]:
        """Execute the generated plan"""
        self.logger.info(f"=== EXECUTION PHASE ===")

        actions = plan.get("actions", [])
        results = []
        document_ids = []
        
        for action in actions:
            action_type = action.get("type")
            
            if action_type == ActionType.RETRIEVAL.value:
                search_query = action.get("query", "")
                self.logger.info(f"RETRIEVAL: Query = \"{search_query}\"")
                retrieval_results = await default_search_retrieval.qdrant_retrieval(
                    query=search_query,
                    collection_name=collection_name
                )
                
                context_text = "\n\n".join([doc.page_content for doc in retrieval_results]) if retrieval_results else "No relevant documents found."
                
                # Collect document IDs for reference
                if retrieval_results:
                    for doc in retrieval_results:
                        if hasattr(doc, 'metadata') and 'document_id' in doc.metadata:
                            if doc.metadata['document_id'] not in document_ids:  # Tránh trùng lặp
                                document_ids.append(doc.metadata['document_id'])
                
                results.append({
                    "action": action,
                    "result": context_text
                })
            
            elif action_type == ActionType.THINK.value:
                # Chain of Thought reasoning
                thinking_query = action.get("query", "")
                self.logger.info(f"THINKING: Query = \"{thinking_query}\"")
                
                reasoning_result = await self._generate_reasoning(
                    thinking_query,
                    context=context,
                    model_name=model_name
                )
                
                self.logger.info(f"THINKING result preview: {reasoning_result[:500]}...")

                results.append({
                    "action": action,
                    "result": reasoning_result
                })
                
        return {
            "original_plan": plan,
            "execution_results": results,
            "document_ids": document_ids
        }
    
    def _create_planning_prompt(self, query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create a planning prompt for the LLM"""
        system_message = {
            "role": "system",
            "content": """You are an intelligent AI assistant that plans how to answer user queries.
Your task is to break down the user's query into a sequence of actions that will lead to the best answer.

Available actions:
1. retrieval - Search for information in the knowledge base
2. think - Reason through a problem step by step
3. generate_response - Create a final response

For each action, provide:
- type: The action type from the list above
- query: What to search for or analyze

Format your response as:
THOUGHT: Reason about the query
PLAN:
1. [Action 1 description]
2. [Action 2 description]
...

ACTIONS:
```json
[
  {
    "type": "action_type",
    "query": "what to search for"
  },
  ...
]
```"""
        }
        
        user_message = {
            "role": "user",
            "content": f"Query: {query}\n\nContext: {context if context else 'No additional context'}"
        }
        
        return [system_message, user_message]
    
    def _parse_actions(self, content: str) -> List[Dict[str, Any]]:
        """Parse actions from LLM response"""
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                import json
                actions = json.loads(json_match.group(1))
                return actions
            except Exception as e:
                self.logger.error(f"Error parsing actions JSON: {str(e)}")
                
        # Fallback: Create action based on original query
        return [{"type": "retrieval", "query": content}]
    
    async def _generate_reasoning(self, query: str, context: Optional[str] = None, model_name: str = "llama3.1:8b") -> str:
        """Generate step-by-step reasoning using Chain of Thought"""
        llm = await self.llm_generator.get_llm(model=model_name)
        
        prompt = [
            {
                "role": "system",
                "content": """You are an analytical expert with comprehensive knowledge.
Think step-by-step about the query, considering all available information to reach a well-reasoned conclusion.
Break down complex problems into simpler components, showing your reasoning process clearly."""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nAvailable information:\n{context if context else 'No additional information'}\n\nThink through this step-by-step:"
            }
        ]
        
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _compile_execution_results(self, execution_results: Dict[str, Any]) -> str:
        """Compile execution results into a single context"""
        compiled = "EXECUTION RESULTS:\n\n"
        
        for result_item in execution_results.get("execution_results", []):
            action = result_item.get("action", {})
            result = result_item.get("result", "")
            
            action_type = action.get("type", "unknown")
            compiled += f"--- {action_type.upper()} ---\n"
            
            if isinstance(result, str):
                compiled += result[:500]  # Limit size
            elif isinstance(result, dict):
                import json
                compiled += json.dumps(result, indent=2)[:500]
            
            compiled += "\n\n"
            
        return compiled
    
    async def generate_final_response(self, 
                                     query: str,
                                     execution_results: Dict[str, Any],
                                     context: Optional[str] = None,
                                     model_name: str = "llama3.1:8b",
                                     thinking_instruction: Optional[str] = None) -> str:
        """Generate the final response based on execution results and reasoning"""
        llm = await self.llm_generator.get_llm(model=model_name)
        
        compiled_results = self._compile_execution_results(execution_results)
        
        prompt = [
            {
                "role": "system",
                "content": f"""You are an intelligent AI assistant that provides clear, helpful responses.
Your task is to generate a comprehensive and well-structured response based on the query and available information.
Include relevant facts and insights from the retrieved documents.
{thinking_instruction or ""}\n"""
            },
            {
                "role": "user",
                "content": f"""Query: {query}

Execution Results:
{compiled_results}

Additional Context:
{context if context else 'No additional context'}

Please provide a comprehensive, well-structured response to the original query based on all information above:"""
            }
        ]
        
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    

    async def generate_streaming_final_response(self, 
                                      query: str,
                                      execution_results: Dict[str, Any],
                                      context: Optional[str] = None,
                                      model_name: str = "llama3.1:8b",
                                      thinking_instruction: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate the final response in streaming mode based on execution results and reasoning"""
        # Get streaming LLM
        streaming_llm = await self.llm_generator.get_streaming_chain(model=model_name)
        
        compiled_results = self._compile_execution_results(execution_results)
        
        prompt = [
            {
                "role": "system",
                "content": f"""You are an intelligent AI assistant that provides clear, helpful responses.
Your task is to generate a comprehensive and well-structured response based on the query and available information.
Include relevant facts and insights from the retrieved documents.
{thinking_instruction or ""}\n"""
            },
            {
                "role": "user",
                "content": f"""Query: {query}

Execution Results:
{compiled_results}

Additional Context:
{context if context else 'No additional context'}

Please provide a comprehensive, well-structured response to the original query based on all information above:"""
            }
        ]
        
        # Use stream_response from LLMGenerator to get chunks
        async for chunk in self.llm_generator.stream_response(streaming_llm, prompt, clean_thinking=True):
            yield chunk


class ChainOfThought(LoggerMixin):
    """
    Implementation of Chain of Thought reasoning to improve LLM responses.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_generator = LLMGenerator()
    
    async def generate_reasoning(self, 
                               query: str,
                               context: Optional[str] = None,
                               model_name: str = "llama3.1:8b") -> Dict[str, Any]:
        """
        Generate step-by-step reasoning for a query
        """
        self.logger.info(f"====================== CHAIN OF THOUGHT ======================")
        llm = await self.llm_generator.get_llm(model=model_name)
        
        reasoning_prompt = [
            {
                "role": "system",
                "content": """You are a reasoning expert. Your task is to think step-by-step about the given query.
Break down your reasoning process into clear logical steps, considering all aspects of the problem.
Analyze the information provided, identify connections, and reach a well-reasoned conclusion."""
            },
            {
                "role": "user",
                "content": f"""Query: {query}

Available Context:
{context if context else 'No additional context is provided.'}

Think step by step to analyze this query:"""
            }
        ]
        
        response = await llm.ainvoke(reasoning_prompt)
        reasoning = response.content if hasattr(response, 'content') else str(response)
        
        self.logger.info(f"CoT reasoning complete: {len(reasoning)} chars")
        self.logger.info(f"Reasoning preview: {reasoning[:1000]}...")

        return {
            "query": query,
            "reasoning": reasoning
        }

class PlanningModule(LoggerMixin):
    """
    Combined planning module that integrates ReAct and Chain of Thought approaches.
    """
    
    def __init__(self):
        super().__init__()
        self.react_agent = ReActAgent()
        self.cot = ChainOfThought()
        
    async def process_query(self, 
                           query: str,
                           collection_name: str,
                           context: Optional[str] = None,
                           model_name: str = "llama3.1:8b",
                           enable_thinking: bool = True) -> Dict[str, Any]:
        """
        Process a query using both ReAct and Chain of Thought
        
        Args:
            query: User's query
            collection_name: Name of vector collection to search
            context: Additional context (e.g. chat history)
            model_name: Name of LLM to use
            enable_thinking: Whether to enable thinking mode
            
        Returns:
            Dict containing response and document_ids
        """
        self.logger.info(f"STEP 1: Creating plan using ReAct...")
        plan = await self.react_agent.plan(query, context, model_name)
        
        self.logger.info(f"STEP 2: Executing plan...")
        execution_results = await self.react_agent.execute_plan(
            plan, 
            collection_name=collection_name,
            context=context,
            model_name=model_name
        )
        
        self.logger.info(f"STEP 3: Creating reasoning using Chain of Thought...")
        compiled_results = self.react_agent._compile_execution_results(execution_results)
        combined_context = f"{compiled_results}\n\n{context}" if context else compiled_results
        
        reasoning_results = await self.cot.generate_reasoning(
            query, 
            context=combined_context,
            model_name=model_name
        )
        
        self.logger.info(f"STEP 4: Generating final response...")
        thinking_instruction = ""
        if "qwen3:14b" in model_name:
            thinking_instruction = "THINKING mode - First THINK carefully before providing your answer." if enable_thinking else "NO-THINKING mode - DO NOT think."
        

        final_response = await self.react_agent.generate_final_response(
            query,
            execution_results,
            context=reasoning_results.get("reasoning", ""),
            model_name=model_name,
            thinking_instruction=thinking_instruction
        )
        
        return {
            "query": query,
            "plan": plan,
            "execution_results": execution_results,
            "reasoning": reasoning_results,
            "response": final_response,
            "document_ids": execution_results.get("document_ids", [])
        }
    

    async def process_query_stream(self, 
                               query: str,
                               collection_name: str,
                               context: Optional[str] = None,
                               model_name: str = "llama3.1:8b",
                               enable_thinking: bool = True) -> Dict[str, Any]:
        """
        Process a query using both ReAct and Chain of Thought with streaming response
        
        Args:
            query: User's query
            collection_name: Name of vector collection to search
            context: Additional context (e.g. chat history)
            model_name: Name of LLM to use
            enable_thinking: Whether to enable thinking mode
            
        Returns:
            AsyncGenerator that streams response chunks and document_ids
        """
        # STEP 1-3: Planning, execution and reasoning are the same as non-streaming
        self.logger.info(f"STEP 1: Creating plan using ReAct...")
        plan = await self.react_agent.plan(query, context, model_name)
        
        self.logger.info(f"STEP 2: Executing plan...")
        execution_results = await self.react_agent.execute_plan(
            plan, 
            collection_name=collection_name,
            context=context,
            model_name=model_name
        )
        
        document_ids = execution_results.get("document_ids", [])
        
        self.logger.info(f"STEP 3: Creating reasoning using Chain of Thought...")
        compiled_results = self.react_agent._compile_execution_results(execution_results)
        combined_context = f"{compiled_results}\n\n{context}" if context else compiled_results
        
        reasoning_results = await self.cot.generate_reasoning(
            query, 
            context=combined_context,
            model_name=model_name
        )
        
        # STEP 4: Use streaming response generation
        self.logger.info(f"STEP 4: Generating streaming final response...")
        thinking_instruction = ""
        if "qwen3:14b" in model_name:
            thinking_instruction = "THINKING mode - First THINK carefully before providing your answer." if enable_thinking else "NO-THINKING mode - DO NOT think."
        
        # Return streaming generator and document_ids
        return {
            "response_stream": self.react_agent.generate_streaming_final_response(
                query,
                execution_results,
                context=reasoning_results.get("reasoning", ""),
                model_name=model_name,
                thinking_instruction=thinking_instruction
            ),
            "document_ids": document_ids
        }