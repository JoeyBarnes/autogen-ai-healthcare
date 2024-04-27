# Automating AI Model Development for Healthcare Using a Multi-Agent Conversational Framework (AutoGen)


**Abstract**

In this project, we harness the power of AutoGen—a cutting-edge generative AI framework—to revolutionize the development and evaluation of AI models in the healthcare domain. Our approach seamlessly integrates contextual research, feature engineering, model development, and model performance evaluation, all driven by a natural language dialog.


**Future Improvements**

- Using RAG for curated healthcare-specific information:
    - Some of the tasks had to be prompt-engineered more prescriptively in order for the agents to be more deterministic (like looking up the ICD codes for a medical diagnosis)
    - https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat/

- Nested agents:
    - https://microsoft.github.io/autogen/docs/notebooks/agentchat_nested_sequential_chats

- Teachable agents: 
    - https://microsoft.github.io/autogen/blog/2023/10/26/TeachableAgent
    - Prompts had to be engineered and tested very carefully in order for the agents to produce consistent results.
    - Having agents that can learn over time when improvements are suggested (either thru reflection and/or additional user feedback).
    - LLM Reflexion
        - https://www.promptingguide.ai/techniques/reflexion
https://arxiv.org/abs/2303.11366

- Self evauluation/criticizing agents
    - i.e. evaluating the auto generated content based on core ethical values (i.e privacy, bias, etc)  in AI
    - https://arxiv.org/abs/2404.12253
    - https://arxiv.org/pdf/2402.09015

- Artificial General Intelligence
    - https://github.com/metamind-ai/autogen-agi


