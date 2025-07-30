from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


# Answer this "{input}" question 
QuestionAnswerTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                        You are an AI assistant designed for accurate information retrieval and question answering. Try to maintain the exact terminology and phrasing from the source material if possible.
                        - Think careful before answer.
                        - Answer this USER QUESTION: "{input}" should be based on "{context}". If you don't know the answer, just say "I couldn't find an answer because the question involves information that has not been documented or is unavailable in the training data." 
                        <|eot_id|>
                         """,
                input_variables=['context', 'input']
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""
                        <|start_header_id|>user<|end_header_id|>
                            - Answer the {input} question strictly based on the given {context}.
                            - Do not rely on external knowledge or make assumptions.
                        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                         """,
                input_variables=['context', 'input']
            )
        )
    ]
)


# QuestionAnswerTemplate = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate(
#             prompt=PromptTemplate(
#                 template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#                 You are an AI assistant designed for accurate question answering. Once relevant documents have been retrieved, your task is to: 
#                 - Try to maintain the exact terminology and phrasing from the source material if possible.

#                 Guidelines:
#                 - **Select & integrate** only the most pertinent information from the retrieved context to answer the user’s question.
#                 - Provide answers that are **direct**, **concise**, and **accurate**.  
#                 - Maintain the exact terminology from the source material when possible
#                 - If the answer does **not** appear in the retrieved context but you are **certain** of the information, answer from your own knowledge.  
#                 - Otherwise, reply: “I couldn't find specific information about this in the available data.”
                                
#                 NOTE: Think careful before answer.

#                 USER QUESTION: "{input}"
#                 AVAILABLE CONTEXT: "{context}"
#                 <|eot_id|>
#                 """,
#                 input_variables=['context', 'input']
#             )
#         ),
#         HumanMessagePromptTemplate(
#             prompt=PromptTemplate(
#                 template="""
#                 <|start_header_id|>user<|end_header_id|>
#                 Answer my question concisely and directly using only the relevant information from the context.
#                 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#                 """,
#                 input_variables=[]
#             )
#         )
#     ]
# )

DirectToLLMTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                        You are an intelligent assistant designed to provide accurate, helpful responses from your own knowledge.
                        
                        Guidelines:
                        - Respond in a conversational, helpful, and engaging manner
                        - Use your extensive knowledge to provide accurate information
                        - When greeted with phrases like "hello", respond with a friendly greeting
                        - When asked about yourself, provide a brief self-introduction
                        - For general questions, provide helpful and informative responses
                        - If you don't know something, be honest about your limitations
                        - Keep responses relevant, natural, and appropriate
                        - Use reasoning to arrive at well-thought-out answers

                        You excel at engaging in thoughtful conversations without needing to retrieve information from external sources.
                        <|eot_id|>
                        """,
                input_variables=[]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""
                        <|start_header_id|>user<|end_header_id|>
                        {input}
                        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        """,
                input_variables=['input']
            )
        )
    ]
)

ContextualizeQuestionHistoryTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""
                    You are a context-aware AI Assistant, dedicated to following instructions precisely without providing any opinions. 
                    Your task is to reformulate the latest user question.
                    Do not rewrite short form of word
                    Ensure the reformulated question is clear, coherent, no yapping and self-contained, providing all necessary context.
                    Your mission is to Formulate the latest User Question into a standalone question that can be understood without the chat history, if necessary, or return it unchanged.
                    IMPORTANT: DO NOT answer the Latest User Question.
                    """,
                input_variables=[]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""
                <The Latest User Question>: {input} 

                Note: 
                - Your mission is to formulate a standalone question.
                - DO NOT answer the question, just reformulate it if needed and otherwise return it as is.
                - No explaination, just return result.
                    
                Standalone question: """,
                input_variables=['input'],
            )
        )
    ]
)

