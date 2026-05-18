# Connect LangChain

## 1-line wrap

```python
from langchain_openai import ChatOpenAI
import agent_lightning as al

llm = ChatOpenAI(model="gpt-4o")
llm = al.wrap_langchain_llm(llm, agent_id="lc_agent")

# Use exactly as before
response = llm.invoke("What is Python?")
```

## Manual tracing with LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
import agent_lightning as al

tracer = al.AsyncTracer(agent_id="lc_agent")

with tracer.run() as run:
    tracer.log_prompt("What is Python?")
    result = chain.invoke({"question": "What is Python?"})
    tracer.log_response(result.content)
    run.add_reward(my_scorer(result.content))
```

## Use optimized prompt in LangChain

```python
tracer   = al.AsyncTracer(agent_id="lc_agent", store=store)
sys_prompt = tracer.get_optimized_prompt("You are a helpful assistant.")

prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human",  "{question}"),
])
chain = prompt | llm
```
