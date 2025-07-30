from typing import List, Dict, Any, Literal, Optional, Annotated
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AnyMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama  
from src.core.utils.config import settings

# === State Definition ===
class AgentState(TypedDict):
    """State for Agent with messages"""
    messages: Annotated[List[AnyMessage], operator.add]
    
    # Metadata for retrieval
    session_id: str
    collection_name: str
    use_multi_collection: bool
    user_id: Optional[str]
    organization_id: Optional[str]
    retrieved_docs: Optional[List[Any]]

    # Control flow
    iteration_count: int
    max_iterations: int


# === Tool Definition ===
@tool
async def search_qdrant(
    query: str,
    collection_name: str = "",
    use_multi_collection: bool = False,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search the Qdrant vector database for relevant documents.
    
    Args:
        query: Search query to find relevant documents
        collection_name: Name of the collection to search in
        use_multi_collection: Whether to search across multiple collections
        user_id: User ID for filtering
        organization_id: Organization ID for filtering
        
    Returns:
        Dictionary containing search results
    """
    return {"content": "Tool not properly initialized"}


class LangGraphChatAgent:
    
    def __init__(self, 
                 model_name: str,
                 provider_type: str,
                 api_key: Optional[str],
                 search_retrieval, 
                 multi_collection_retriever,
                 enable_thinking: bool = False,
                 max_iterations: int = 3):  # Limit iterations
        
        self.search_retrieval = search_retrieval
        self.multi_collection_retriever = multi_collection_retriever
        self.enable_thinking = enable_thinking
        self.max_iterations = max_iterations
        
        # Create LangChain model
        self.model = self.create_langchain_model(model_name, provider_type, api_key)
        
        # System prompt
        self.system = """You are an intelligent assistant with access to a knowledge base through the search_knowledge tool.

When to use the search_knowledge tool:
- When asked about specific documents, data, or information that might be in the database
- When you need factual information beyond your training data
- When the question references specific topics that require retrieval

When NOT to use the tool:
- For general greetings or simple conversational responses
- When you already have the information from previous tool calls
- For calculations or logical reasoning

IMPORTANT: After using the tool, analyze the results and provide a comprehensive answer. Do not call the tool multiple times for the same query."""
        
        # Build graph
        self.graph = self._build_graph()
    
    @staticmethod
    def create_langchain_model(model_name: str, provider_type: str, api_key: Optional[str] = None):
        if provider_type == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    def _build_graph(self):
        """Build agent graph with proper flow control"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("llm", self.call_llm)
        graph.add_node("tools", self.execute_tools)
        
        # Add conditional edges với iteration check
        graph.add_conditional_edges(
            "llm",
            self.route_after_llm,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Tools always go back to LLM
        graph.add_edge("tools", "llm")
        
        # Set entry point
        graph.set_entry_point("llm")
        
        # Compile
        return graph.compile()
    
    def route_after_llm(self, state: AgentState) -> Literal["tools", "end"]:
        """Route sau khi LLM response"""
        # Check iteration limit
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"⚠️ Reached max iterations ({self.max_iterations}), stopping")
            return "end"
        
        # Check for tool calls
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    async def call_llm(self, state: AgentState) -> Dict[str, Any]:
        """Call LLM with system prompt and tool binding"""
        messages = state["messages"]
        
        # Add system message if not already there
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system)] + messages
        
        # Create inline search tool to avoid recursion
        search_tool = self._create_search_tool_simple()
        
        # Bind tools to model
        model_with_tools = self.model.bind_tools([search_tool])
        
        # Invoke model
        response = await model_with_tools.ainvoke(messages)
        
        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        
        return {
            "messages": [response],
            "iteration_count": iteration_count
        }
    
    def _create_search_tool_simple(self):
        """Create simple search tool without closure issues"""
        @tool
        async def search_knowledge(query: str) -> str:
            """
            Search the knowledge base for relevant information.
            
            Args:
                query: Search query to find relevant documents
            """
            return f"[Tool will search for: {query}]"
        
        return search_knowledge
    
    async def execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """Execute tool calls with actual retrieval"""
        tool_calls = state["messages"][-1].tool_calls
        results = []
        
        for tool_call in tool_calls:
            print(f"🔧 Executing tool: {tool_call['name']} with query: {tool_call['args'].get('query', '')}")
            
            if tool_call["name"] == "search_knowledge":
                # Get search query
                query = tool_call["args"].get("query", "")
                
                # Extract metadata from state
                collection_name = state.get("collection_name", "")
                use_multi_collection = state.get("use_multi_collection", False)
                user_id = state.get("user_id")
                organization_id = state.get("organization_id")
                
                try:
                    # Perform actual retrieval
                    if use_multi_collection and user_id:
                        docs = await self.multi_collection_retriever.retrieve_from_collections(
                            query=query,
                            user_id=user_id,
                            organization_id=organization_id,
                            top_k=5
                        )
                    else:
                        docs = await self.search_retrieval.qdrant_retrieval(
                            query=query,
                            collection_name=collection_name,
                            top_k=5
                        )
                    
                    if docs:
                        # Format context
                        context = "\n\n---\n\n".join([
                            f"Document {i+1}:\n{doc.page_content}" 
                            for i, doc in enumerate(docs[:3])  # Limit to 3 docs
                        ])
                        
                        # Store docs for reference
                        state["retrieved_docs"] = docs
                        
                        result_content = f"Found {len(docs)} relevant documents. Here are the top results:\n\n{context}"
                    else:
                        result_content = "No relevant documents found in the knowledge base for this query."
                    
                except Exception as e:
                    result_content = f"Error searching database: {str(e)}"
                    print(f"❌ Search error: {str(e)}")
                
                # Create tool message
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    content=result_content
                )
                results.append(tool_message)
            else:
                # Unknown tool
                tool_message = ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    content="Unknown tool"
                )
                results.append(tool_message)
        
        return {"messages": results}
    
    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke agent with state"""
        # Initialize iteration count
        state["iteration_count"] = 0
        state["max_iterations"] = self.max_iterations
        
        # Run graph
        result = await self.graph.ainvoke(state)
        
        # Extract final response
        final_message = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                final_message = msg
                break
        
        if not final_message:
            final_message = result["messages"][-1]
        
        # Clean thinking if needed
        content = final_message.content
        if not self.enable_thinking and "<thinking>" in content:
            import re
            content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        
        return {
            "response": content,
            "messages": result["messages"],
            "retrieved_docs": result.get("retrieved_docs", [])
        }