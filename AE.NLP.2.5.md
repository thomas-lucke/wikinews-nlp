## Sprint: Text analysis using NLP

### Part 5: NLP project

Let’s assume you are working in a company which relies on AI-driven programs to automate manual tasks. The company aims to acquire named entity recognition (NER) to quickly make its way into business operations. One of these operations is to find and extract important information (described by “entities”) across the news and analyze them in terms of brand health, business competitors, customer experience, social events, and but not least, political current affairs. 
 
With this task, you are taking the role of data scientist who works with Wikinews data (https://github.com/PrimerAI/WikiNews-multilingual) which covers more than 15 thousand articles in 33 languages. Wikinews articles are regular news articles containing actual information about political events, social life, technologies and more. You, as a data scientist, need to select the 3-4 areas (subsets of raw data) you are interested in, and extract and analyze entities corresponding to these areas, summarize a selected bunch of these articles by keeping the most relative key facts, and predict the most appropriate topic for given texts. Remember, your findings and insights should focus on particular topics, activities or events which reflect real life rhythm and trends. 

News articles could be quite long and complex so it can be a challenge to process big amounts of this information by extracting the most relevant key points. This is where text summarization comes into play. This technique reduces the time required for grasping lengthy pieces such as news articles without losing vital information. For this project, you will grab a bunch of texts from multiple 3-4 categories of news data and summarize them with the purpose of reducing the time and effort required to read and understand lengthy news texts as well as to ensure the accuracy and completeness of a summary.

Once you have summarized texts, one of the best ways to check how accurate they are is to calculate similarity scores between them and  actual ones. If the calculated similarity score between summarized and original text is high (e.g. > 0.8), suppose that the summarized text keeping the main information and is suitable for the business.

### Objectives

Your main goal in this task is to deliver a comprehensive overview of selected categories of news articles in specific country(-ies), by implementing named entity recognition (NER), text summarization and analysis of similarities between original and summarized articles. For NER part, you need to analyze the most frequently mentioned “named entities” (furthermore “NE”) revealing its dynamic over time and providing an aggregated view on recognised NE. For this purpose you can use a selected NER framework such as Spacy or DeepPavlov. Please remember, that suggested NER frameworks work on multi-language datasets, so you can experiment with multiple language texts in the given data.

To address the ever-growing amount of text data available online it is very important to summarize a large amount of text to discover relevant information and to consume relevant information faster than using human efforts only. As you are a data scientist working on a news dataset, you aim to summarize these texts containing multiple topics. Select 3 or more categories with 10-20 articles per category and apply extractive or abstractive summarization techniques. 

Check for grammar, style errors and provide your findings on these criterions. Once you have summarized versions of texts, you can evaluate how they are similar with the actual texts. To calculate the similarity score you need to enhanche word embedding techniques that transform words into dense vectors where semantically similar words are close in the vector space. 

### Requirements

1. Apply text pre-processing: Break down the raw text into manageable units and identify the grammatical roles of words;
2. Perform Named entity recognition: each entity has to be associated with news articles and corresponding metadata;
3. Investigate wrongly predicted entities: While NER has made a lot of progress for languages like English, it doesn’t have the same level of accuracy for many others. This is often due to a lack of labeled data in these languages;
4. Perform text summarization for selected articles. In order to make long texts shorter and still usable for the business, you should apply selected text summarization techniques and summarize 10-20 articles for each of selected 3 or more categories;
5. Investigate text similarity scores. Calculate similarities scores between summarized texts and actual ones. Visualize the distribution of similarities score and explain for which texts similarities score are the lowest, and for which ones the similarity score is the highest.

### Evaluation Criteria

- Selected NER framework applied on pre-processed data and generated meaningful results including named entities, its categories and related metadata.

- The NER analysis of the results should include at least two selected criterions, e.g. aggregated overview, dynamic over time, clustered to semantic groups, etc.

- Calculated similarities scores between summarized and actual texts among different news categories should be visualized and explained.

- Predicted topics for selected set of text are logical and reflect the actual topics or categories.

- The code should meet PEP8 coding standards, logically organised with functions and/or OOP classes.
