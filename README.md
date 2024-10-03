# Week 3 Deliverables

# Day 1 & 2: Advanced LangChain Chains and Custom Agent

## Deliverables
`multi_chain.py`,  `agent.py`

## Building Custom Agent and Tools
I have built an LLM architecture that retrieves information from both Wikipedia and Stack exchange and then summarizes this content. This feature is useful for information retrieval, increasing reliability of information.

## External API Integration
### Before
```
The case "FDA v. Alliance for Hippocratic Medicine" is a significant legal battle that occurred in 2023-2024 concerning access to mifepristone, a medication used for medication abortions and miscarriage management in the United States.

Key points of the case include:

1. Challenge to FDA approval: The Alliance for Hippocratic Medicine, along with other anti-abortion groups, challenged the FDA's approval of mifepristone, which had been in use for over 20 years.

2. Lower court rulings: In April 2023, a federal judge in Texas ruled to suspend the FDA's approval of mifepristone. This decision was partially stayed by the 5th Circuit Court of Appeals.

3. Supreme Court involvement: The case reached the Supreme Court, which issued an order in April 2023 to maintain the status quo, allowing continued access to mifepristone while legal proceedings continued.

4. Arguments: The plaintiffs argued that the FDA's approval process was flawed and that the drug was unsafe. The FDA and its supporters maintained that mifepristone is safe and effective, backed by years of data and use.

5. Implications: The case has significant implications for reproductive healthcare access, FDA authority, and abortion rights in the United States.

6. Ongoing nature: As of my last update in April 2024, the case was still ongoing, with potential for further legal developments.

This case has been closely watched due to its potential impact on abortion access and pharmaceutical regulation in the United States. If you'd like more detailed information on any aspect of this case, please let me know.
```

### After
```
Based on the information I\'ve found, I can provide you with a summary of "FDA v. Alliance for Hippocratic Medicine":\n\nFDA v. Alliance for Hippocratic Medicine was a significant United States Supreme Court case decided in 2024. The case centered around a challenge to the U.S. Food and Drug Administration (FDA)\'s approval of mifepristone, a drug commonly used in medical abortion procedures.\n\nKey points of the case include:\n\n1. The plaintiffs, led by the Alliance for Hippocratic Medicine (AHM), argued that the FDA had not properly approved mifepristone for pregnancy termination under Federal Food, Drug, and Cosmetic Act regulations.\n\n2. The case came in the wake of the 2022 Dobbs v. Jackson Women\'s Health Organization decision, which overturned Roe v. Wade and eliminated the federal constitutional right to abortion.\n\n3. The legal journey was complex:\n   - A district judge initially issued a preliminary injunction suspending mifepristone\'s approval.\n   - The Fifth Circuit Court partially reversed this decision, allowing the drug\'s 2000 approval to stand but putting on hold changes to distribution rules made in 2016.\n   - A separate federal judge issued an order to maintain mifepristone distribution in 16 states and Washington, D.C.\n\n4. The case ultimately reached the Supreme Court, which ruled unanimously on June 13, 2024.\n\n5. The Supreme Court\'s decision:\n   - Ruled that the Alliance for Hippocratic Medicine did not have standing to bring the case under Article III of the Constitution.\n   - Neither AHM nor the groups it represented had shown injury.\n   - Reversed the lower court decisions.\n   - Restored mifepristone\'s availability under the current FDA rules.\n\nThis case was significant in the ongoing legal battles surrounding abortion access in the United States following the Dobbs decision, and it reaffirmed the FDA\'s authority in approving and regulating medications like mifepristone.'
```

## Impact and Potential use cases for LLM Agents
LLM Agents are a great tool for modeling architecture, especially for customer service chatbots because they are able to better serve customer needs either through access to company's internal data, or integrating external api to provide more information. 

For example, a customer may want to know more about places they can go for sightseeing; LLM agents can decide to gather location information as well as weather data to recommend such outputs. 

Of course, this ties in with knowledge retrieval systems where LLM agents when instructed to make informed or domain specific decision can refer to authoritative documents to guide their decision making. For legal and healthcare this is especially important to assert that the information is backed by some sources of truth.

LLM agents under the hood are just making structured outputs that determines whether or not they should utilize tools. LLM agents are better recommended to be constructed using LangGraph as the abstraction allows for more sophisticated architecture to suit most requirements that would otherwise be complex in Langchain. 

# Day 3 & 4: Langchain for Multi-modal Applications
## Deliverables
`multi_modal.py` 
## Multi-modal Integration with LangChain
LangChain provides support for multi-modal integration like inserting pdfs and images into closed source large language models like OpenAI ChatGPT and Anthropic's Claude. These models can provide pdfs and images but can lack integration with audio files. 

In this case, LangChain can provide a comfortable workflow to allow a language model that can support audio inputs and have the Q&A model answer the question based on the outputs of that second model.

```
Based on the analysis of the image, I can describe what's happening in this picture:  
  
The image captures an intense moment during an NBA basketball game between the Washington Wizards and the Indiana Pacers. Here's what's happening:  
  
1. Player confrontation: The picture shows three players in a heated moment of gameplay. Two players from the Washington Wizards (in white and red uniforms) are defending against one player from the Indiana Pacers (in navy blue and gold).  
  
2. Ball possession: The Pacers player is at the center of the action, trying to drive towards the basket while maintaining control of the basketball. You can see the ball in his hands.  
  
3. Defensive effort: The two Wizards players are actively trying to stop the Pacers player's advance. One Wizards player (jersey number 30) is applying pressure from the left side, while his teammate (number 3) is reaching in from the right to try and disrupt the ball handler or potentially steal the ball.  
  
4. Physical intensity: All three players show signs of intense physical exertion. Their muscles are visibly tensed, and their bodies are leaning into the action, demonstrating the physicality of the sport.  
  
5. Competitive atmosphere: The image captures the essence of professional basketball's competitiveness, showcasing the struggle for ball possession and court position.  
  
6. Setting: The background reveals the basketball court and hints at a crowded arena, with some blurred spectators visible in the stands, indicating this is a professional-level game in a major arena.  
  
This snapshot effectively freezes a split-second of high-stakes competition, highlighting the speed, skill, and physicality involved in NBA basketball. It's a great example of the kind of defensive pressure and offensive determination that makes basketball such an exciting sport to watch.
```

## Uses of Multi-modal contents
### E-commerce
There are many possibilities in which e-commerce applications can benefit. 

The first is customer experience. Users can upload photos of a particular product and the response would the listing of the product. Audio could be used to increase accessibility and recommendation of products

Provide summarization of manuals that highlight important parts of how the product can be used.

There is also a possibility for vendors to integrate automated customer service through LLM agents that would collect all the necessary product defect or dissatisfaction through images and text and have the LLM act accordingly.

Enhancing Content moderation to ensure that images, text are appropriate for users.

### Healthcare
There are many uses for multi modal in healthcare but they require state of the art tools to make analysis and diagnosis compelling to be deployed.

1. you could integrate a ML model dedicated for a more domain specific use, like medical imaging, with an LLM architecture that would interpret those results.
2. Summarization of patient's medical history that includes parsing necessary documents and other health concerns.
3. Digitization of handwritten notes using either a Vision based LLM or OCR, integration in the LLM Agent architecture.
4. Increased accessibility of healthcare use through the use of audio inputs to interface with services.
5. Even if some of these functions require a non-transformer architecture to perform the best results, integration with LLM systems will allow the patients to interact with agents that perform best in specific task; LLM then perform actions to satisfy the needs of patients.

### Education
The possibilities of enhancing education is possible through the use of LLMs as a medium for learning. 

Multi-modal architecture allows students to not just expose themselves to interacting with text, but also through audio inputs and outputs. The expectation is a simulation of a classroom environment with natural sounding voices.

Another possible use case is increasing interactivity of textbooks and explaining the process through an auditory output and as well as inputs. Generating images and graphs from textbook examples is a good way for students to effectively learn their materials.

There is a controversial possibility of assessment of student performance through the collection of documents like report card and answer sheets. Teachers can customize how they want to grade their stuff and many people will follow.

# Day 5 & 6: LangChain for LLM fine-tuning and Customization
## Deliverables
`finetune/*`, `finetune_integr_app.py`

## Technical Specification
I will be using the latest LLM model Llama 3.2 3B for fine tuning the model and the amazing part of it is is the 128k context window, which is remarkable for handling large documents at this particular size. This model especially is great for things like tool calling which integrates well with LangChain and constructing agents.

## Fine Tuning 
The fine tuning process will be done on `finetune/finetuning.ipynb` and training process will be done in Google Colaboratory.

## Custom Application

## Fine Tuning process review
### Benefits
1. If you have a specific requirements or some specific structure of output that other base models cannot provide, then fine tuning it and integrating it as an llm agent allows it to be more specialized in performing certain tasks.
	1. Fine tuning allows better outputs than the pre-trained models
	2. Langchain or Langgraph allows developers to create architectures that utilize mixture of specialized models through a workflow model.
3. Fine tuning is preferable to Retrieval methods to pass in prompts if there is ample of domain specific terminology that retrieval methods would constantly have to fetch from some information, increasing the prompt size and reducing the performance of inference.
	1. Langchain provides retrieval methods when available
4. Fine tuning allows smaller models to have comparable performance to larger model on niche applications
	1. We can have multiple fine-tuned models, each with a specialized use case without having to use a proprietary model or a massive model. Langchain and Langgraph provide the tools for integrating these fine-tuned models for business needs.
5. Potential Cost Savings: 
	1. We can probably run a collection of fine tuned models with a good a comparable performance to proprietary models as well as just having to use smaller models: which can be expensive as inference computation is bill metered.
### Challenges
1. Fine tuning process requires steps to understand specific use case does the base model lack and how fine tuning it can potentially improve performance.
2. Fine tuning process is tedious if you do not have a good training environment: good gpu, cpu, etc. 
3. Choosing a good dataset is the most important and difficult part of the process as they can determine the quality of the fine tuned model
4. Expertly fine tuning needs domain knowledge to efficiently squeeze all the performance you can get.
5. Overfitting is a possible concern 
## Challenges in completing Deliverable
1. So I think that in fine-tuning Llama 3.2 model I have failed to consider which base model is fitting to be fine-tuned or if needed at all.
2. Fine tuning the model also poses a problem as Llama 3.2 is able to handle tool calling while the fine tuning model lost most of its tool calling capabilities: harming Agent and tool integration.
3. Fine tuning an effective model would take few days at best but probably a week for most effective use.
4. In production I would have probably used a fine-tuned SQL specialized model to handle specialized SQL query generation. If the fine-tuned SQL model does not handle tool-calling, I would have fine tune on top of this model or used this SQL model as a tool itself.

# Day 7 & 8: LangChain for Knowledge Base and Document Summarization
## Deliverables
`summarization.py`

## Supported Documents
1. pdf
2. articles
3. reports

## Specific Domains
1. Research Paper
2. legal contracts
3. business reports

## How should an App interact with this?
1. We are going to use FastAPI to handle file upload into the server's file system.
	1. In production this would probably be done somewhere else
## What should be the output format?
Summarization in a generalized sense

## Outline
![[Screenshot 2024-10-02 at 2.35.25 PM 3.png]]
# Deliverables: LLM Performance of Summaries
Here are the models that will be used to assess the performance of summarizing each documents.
1. Claude
2. Llama 3.2 3B
3. Mistral 4B
4. Gemma 2 2B

#### Claude
1. Probably the best part of the summarization as it doesn't go into the finer details and goes top-level with the summarization
```
'This paper discusses paths to equilibrium in games, focusing on multi-agent reinforcement learning (MARL) and game theory.  
The authors study sequences of strategies satisfying a pairwise constraint inspired by policy updates in reinforcement learning, where an agent who is best responding in one period does not switch its strategy in the next period.  
They call these sequences "satisficing paths" and explore their properties.\n\n  
Key points:\n\n1. The paper introduces the concept of satisficing paths as a generalization of  
best response paths in game theory.\n\n2. Satisficing paths are defined as sequences of strategy profiles  
where players only update their strategies if they are not currently best responding.\n\n  
3. The authors prove that every finite normal-form game has the satisficing paths property,  
meaning there exists a satisficing path from any initial strategy profile to a Nash equilibrium.\n\n4. This result has implications for the design of MARL algorithms, suggesting that algorithms based on satisficing updates can converge to equilibrium.\n\n5. The paper extends the analysis to Markov games and discusses potential applications to decentralized learning scenarios.\n\n6. The authors highlight connections between their work and existing concepts in game theory and reinforcement learning, such as fictitious play and no-regret learning.\n\n7. The proof techniques developed in the paper may be useful for analyzing other types of strategy update processes in games.\n\n8. The paper concludes by discussing open questions and future research directions, including extensions to more complex game settings and the development of new MARL algorithms based on satisficing principles.\n\nThis work contributes to the theoretical understanding of learning dynamics in multi-agent systems and provides insights that may inform the design of more effective MARL algorithms.\n\n[End of Notes, Message #1]'}
```

#### Llama 3.2b
1. Seems to miss the main mark of the paper. 
2. "lemma1 " or "theorem 1" are not defined.
4. What is $\lambda$ ?
5. In general seems to have a narrow focus in its summarization
```
'This text is a continuation of the previous discussion on Markov games  
and their comparison to normal-form games.  
The author presents a challenge in extending a proof technique from normal-form games to Markov games.\n\n  
In essence, the author identifies two main limitations:\n\n  
  
  
1. **Characterization of indifference conditions**:  
In normal-form games, indifference conditions can be characterized  
by the roots of a polynomial in the parameter λ.  
However, in Markov games, this is not generally true.\n  
  
2. **Proof technique**: The proof technique used to prove Lemma 1 relies on the explicit form of the indifference condition in stateless, finite normal-form games.\n\n  
The author then presents a counterexample showing that the proof technique does not generalize to Markov games without modifications. Specifically:\n\n* In Markov games, the indifference condition cannot be characterized by the roots of a polynomial in λ.\n  
* The proof technique relies on the fact that the coefficient terms do not depend on the strategy zλ and hence do not depend on λ.\n\n  
As a result, the author concludes that while the remaining elements of the proof of Theorem 1 carry over to Markov games, the characterization of indifference conditions and the proof technique need to be modified.  
However, the author leaves open the possibility of generalizing Theorem 1 to Markov games by finding alternative techniques for characterizing indifference conditions and proving the theorem.\n\n  
The text concludes with a note that although this limitation is a challenge, it does not pose a fundamental obstacle to extending Theorem 1 to Markov games.'
```

### Mistral
1. Lemma 1 and theorem 1 are defined.
2. Should not refer to abstract figures like lemma or theorem, rather complete ideas
3. too low level and less structured
```
" This text discusses the analysis of indifference conditions in Markov games, which are games where the outcome depends on the sequence of states and actions taken by all players, unlike finite normal-form games where outcomes depend solely on the strategies chosen by each player.\n\nThe text explains that the optimality condition for a single agent (player i) best responding to a policy π−i, which is turned into an indifference condition between two actions ai 1 and ai 2 in state s as follows:\n\nQ∗i(π−i(s, ai 1)) = Q∗i(π−i(s, ai 2)) = max ai Q∗i(π−i(s, ai)).\n\nThe text then introduces a one-parameter family of policies {̟λ : 0 ≤ λ leq 1} parameterized by some λ ∈ [0, 1], and notes that the indifference condition cannot generally be characterized by the roots of a polynomial in the parameter λ. The text also mentions that this limitation does not pose a fundamental obstacle to the generalization of Theorem 1 (which states that best responses are unique) but that it requires a different proof for Lemma 1 (which is key to the proof of Theorem 1 in finite normal-form games).\n\nFinally, the text suggests that one can generalize Theorem 1 to apply to Markov games by applying similar machinery as used in this paper but substituting a different proof for that of Lemma 1. The text leaves as an open question for future research how to find such a proof.\n\nIn addition, the text mentions a special case where the indifference condition does yield a polynomial, which is when player i's action does not influence transition probabilities. In this case, the normal-form proof technique can be applied without modification."
```

# Uses cases in industries
### Legal
1. Although summarization should not be used in court presented to a judge, it should however aid lawyers in reading stacks of court opinions and law documents that could potentially be relevant. Then if a document seems promising enough, the lawyer will go in-depth to analyze whether that document is a good idea.
2. For laymen, having to summarize key points of a contract depending on their preferences on privacy, obscure clauses, etc. This is why having a robust summarization LLM is important to capture the essence of the texts.
3. In general
	1. Speed up process of analyzing precedents, court opinions in common law.

### Finance
1. Summarization of annual financial reports are crucial to extract key metrics
	1. Take a SEC filings and Financial Reports
	2. Extract metrics
		1. financial metrics
		2. risk factors
		3. growth
2. Summarization of industry reports aids market research
	1. Providing tools for insight baked into summarization output
	2. Aggregate industry reports for identifying trends and growth opportunities
3. Summarization of news help inform trading decisions
	1. Key government documents can help forecast the health of the economy/market

### Academia
1. Research:
	1. For research, the purposes of summarization of research documents could help comprehend the document as a whole. The readers will grasp the main concepts of the paper that the abstract will fail to mention.
	2. Can help create abstracts based on other sections of the research paper.
2. undergraduates and graduates:
	1. Summarization of textbooks will allow students to comprehend difficult piece of information


## Evaluation LangChain
LangChain provides the necessary tools for creating agents and tool calling, which can increase the workflow summarization tasks. For Creating Agent Architecture, combining many tools, LLMs: LangChain provides the tools for such summarization workflows.
