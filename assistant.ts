import { WEBSITEANALYSTAUDIENCE } from './prompts'
import { loadQAStuffChain } from 'langchain/chains'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import {
  ChatPromptTemplate, PromptTemplate
} from '@langchain/core/prompts'
import dayjs from 'dayjs'
import { ChatAnthropic } from '@langchain/anthropic'
import {
  RunnableSequence, RunnableWithMessageHistory
} from '@langchain/core/runnables'
import { pull } from 'langchain/hub'
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { StructuredOutputParser } from 'langchain/output_parsers'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import fetch from 'node-fetch'
import { BufferMemory } from 'langchain/memory'
import fs from 'fs'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import path from 'path'
import type { ChainValues } from '@langchain/core/utils/types'
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis'

import { TavilySearchResults } from '@langchain/community/tools/tavily_search'
import { createRetrieverTool } from 'langchain/tools/retriever'
import { ChatMessageHistory } from 'langchain/stores/message/in_memory'
import { v4 as uuidv4 } from 'uuid'
import { z } from 'zod'

import { supabase } from '../../services/supabase'
import huggingface from './huggingface'

interface GenerateWebsiteAnalysisResponse {
  keywords: string[]
  audience: string
}

export const openai = {
  evaluateWebsite: async (
    url: string
  ): Promise<GenerateWebsiteAnalysisResponse> => {
    if (!/^https?:\/\//i.test(url)) {
      url = 'http://' + url
    }

    const model = new ChatOpenAI({
      modelName: 'gpt-4o',
      temperature: 0.6,
      maxConcurrency: 10
    })

    const loader = new CheerioWebBaseLoader(url)

    const documents = await loader.load()

    const chain = loadQAStuffChain(model)

    const analysisChain = await chain.call({
      input_documents: documents,
      question: WEBSITEANALYSTAUDIENCE
    })

    const parser = StructuredOutputParser.fromNamesAndDescriptions({
      keywords: 'Top SIX high intent SEARCH keywords that best describe the website, comma separated.',
      audience: 'A short but detailed description of the audience of the website.'
    })

    const parseChain = RunnableSequence.from([
      PromptTemplate.fromTemplate('{input}\n{format_instructions}'),
      model,
      parser
    ])

    const response = await parseChain.invoke({
      input: analysisChain.text,
      format_instructions: parser.getFormatInstructions()
    })

    return {
      keywords: response.keywords.split(', '),
      audience: response.audience
    }
  },
  generateLongtailKeywords: async ({
    subject
  }): Promise<{ keywords: string[] }> => {
    const system = `Please ignore all previous instructions. Please respond only in the english language. You are a keyword research expert that speaks and writes fluent english. I want you to generate a list of 10 long-tail keywords for the given subject. please print "List of same keywords separated by commas:". On the next line print the same list of keywords at the bottom separated by commas. Do not repeat yourself. Do not self reference. Do not explain what you are doing.
    `
    const output = z.object({
      keywords: z.array(z.string())
    })

    const model = new ChatOpenAI({
      modelName: 'gpt-4o',
      temperature: 0.6,
      maxConcurrency: 10
    })

    const structuredLlm = model.withStructuredOutput(output)

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', `${system}`],
      ['human', '{input}']
    ])

    const chain = prompt.pipe(structuredLlm)

    const respone = await chain.invoke({
      input: subject
    })

    return respone
  },
  translateArticle: async (
    content: string,
    targetLang: string,
    spaceModel: string
  ): Promise<string> => {
    const model = await openai.modelSelector(spaceModel)

    const prompt = ChatPromptTemplate.fromMessages([
      ['user', `Translate the following article to {language}:
      {content}
      `]
    ])

    const chain = prompt.pipe(model)
    const result = await chain.invoke({
      language: targetLang,
      content
    })

    // @ts-expect-error - I don't know why this is happening
    return result.content
  },
  modelSelector: async (
    model?: string
  ) => {
    if (!model) {
      return new ChatOpenAI({
        modelName: 'gpt-4o',
        temperature: 0
      })
    }
    if (model.includes('claude')) {
      return new ChatAnthropic({
        model: model || 'claude-3-5-sonnet-20240620',
        apiKey: process.env.ANTHROPIC_API_KEY,
        maxTokens: 4096
      })
    }
    return new ChatOpenAI({
      modelName: model || 'gpt-4o',
      temperature: 0.6,
      maxConcurrency: 10,
      maxTokens: 4096
    })
  },
  polishArticle: async ({
    space,
    content,
    split,
    removeTransitions,
    midarticle,
    repitition,
    simplification,
    activeVoice,
    customPrompt
  }: {
    space: any
    content: string
    split: boolean
    removeTransitions: boolean
    midarticle: boolean
    repitition: boolean
    simplification: boolean
    activeVoice: boolean
    customPrompt: string
  }) => {
    const llm = await openai.modelSelector(space.model as string)

    const messageHistory = new ChatMessageHistory()

    const searchTool = new TavilySearchResults()

    const tools = [searchTool]

    const prompt = await pull<ChatPromptTemplate>(
      'hwchase17/openai-functions-agent'
    )

    const agent = await createOpenAIFunctionsAgent({
      // @ts-expect-error - I don't know why this is happening
      llm,
      tools,
      prompt
    })

    const agentExecutor = new AgentExecutor({
      agent,
      tools
    })

    const agentWithChatHistory = new RunnableWithMessageHistory({
      runnable: agentExecutor,
      getMessageHistory: (_sessionId) => messageHistory,
      inputMessagesKey: 'input',
      historyMessagesKey: 'chat_history'
    })

    const result = await agentWithChatHistory.invoke({
      input: `You are a copywriting expert tasked with improving and polishing an article. The article is written in markdown format. You need to do the following:
      ${split ? '- Split up paragraphs into 1-3 sentences' : ''}
      ${midarticle ? '- Remove mid-article conclusions' : ''}
      ${removeTransitions ? '- Remove transitions' : ''}
      ${repitition ? '- Reduce repetition' : ''}
      ${simplification ? '- Simplify complex sentences' : ''}
      ${activeVoice ? '- Convert passive voice to active voice' : ''}
      ${customPrompt ? `- ${customPrompt}` : ''}

      return the polished article in markdown format, READY TO UPLOAD AND NOTHING ELSE

      Here is the article:
      ${content}`
    }, {
      configurable: {
        sessionId: 'polishArticle'
      }
    })

    const output = result.output.replace(/^`/, '').replace(/`$/, '')

    return output
  },
  titleAndSummaryFromSearch: async ({
    results,
    space,
    excludedSlugs
  }: {
    results: any
    space: any
    excludedSlugs: string[]
  }) => {
    const llm = await openai.modelSelector(space.model as string)

    const messageHistory = new ChatMessageHistory()

    const searchTool = new TavilySearchResults()

    const tools = [searchTool]

    const prompt = await pull<ChatPromptTemplate>(
      'hwchase17/openai-functions-agent'
    )

    const agent = await createOpenAIFunctionsAgent({
      // @ts-expect-error - I don't know why this is happening
      llm,
      tools,
      prompt
    })

    const agentExecutor = new AgentExecutor({
      agent,
      tools
    })

    const agentWithChatHistory = new RunnableWithMessageHistory({
      runnable: agentExecutor,
      getMessageHistory: (_sessionId) => messageHistory,
      inputMessagesKey: 'input',
      historyMessagesKey: 'chat_history'
    })

    const result = await agentWithChatHistory.invoke({
      input: `Given this list of search results: ${results.map((result: any) => result.link).join('\n')}

      Create a unique title, summary, and a list of links to other articles that are relevant to the original article.

      Create a unique slug for the article.

      ${excludedSlugs.length > 0 && `Do not use the following slugs: ${excludedSlugs.join(', ')}`}
      
      - ${space.audience && `The audience you are writing for is: ${space.audience}`}`
    }, {
      configurable: {
        sessionId: 'clusterResult'
      }
    })

    const outputSchema = z.object({
      title: z.string().describe('The title of the article'),
      summary: z.string().describe('The summary of the article'),
      links: z.array(z.string()).describe('The links to the other articles'),
      slug: z.string().describe('The slug of the article')
    })

    const model = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
      temperature: 0
    })

    const structuredLlm = model.withStructuredOutput(outputSchema)

    const output = structuredLlm.invoke(result.output as string)

    return await output
  },
  generateArticleOutline: async ({
    // Space with the persona
    space,
    links,
    title,
    summary,
    internalLinks,
    progressId
  }: {
    space: any
    links?: string[]
    title?: string
    summary?: string
    internalLinks?: string[]
    progressId: string
  }): Promise<string | null> => {
    if (!space) {
      throw new Error('Space does not have a persona')
    }

    const agentWithChatHistory = await openai.createAgent({
      space,
      links,
      progressId
    })

    await agentWithChatHistory.invoke({
      input: `Create a Title and thesis statement for the for an article I am writing:

      ${summary ? `Here is the summary of the article: ${summary}` : '- Use the original article to create a summary'}
      ${title ? `The original title of the article is: ${title}` : ''}
      ${links?.length ? `Use the these articles, as a reference when writing the Title and thesis statement: ${links.join(', ')}` : ''}
      ${internalLinks?.length ? `Add these internal links to the article: ${internalLinks.join(', ')}` : ''}
  
      When writing the new title and thesis statement, please follow these guidelines:
      Write in the ${space.voice}
      ${space.audience && `The audience you are writing for is: ${space.audience}`}
      ${space.keywords && space.keywords.length > 0 && `If relevant, it should include the following keyword if it makes sense, don't force keywords: ${space.keywords.join(', ')}`}
      ${space.customTonePrompt ? space.customTonePrompt : `Mimic the ${space.persona.tone} style of ${space.persona.name as string} throughout all writing. DO NOT MENTION OR IMPLY THAT YOU ARE ${space.persona.name}, because you are JUST MIMICKING their personality & style.`}`
    }, {
      configurable: {
        sessionId: progressId
      }
    })

    if (space.realTime) {
      await supabase.from('RewriteProgress').update({
        rewriteId: progressId,
        progress: 'Adding external links'
      }).match({ rewriteId: progressId })
      await agentWithChatHistory.invoke({
        input: 'Using your access to Google Search Automatically external links using relevant anchor text to support the article, that should be used to support the article.'
      }, {
        configurable: {
          sessionId: progressId
        }
      })
    }

    await supabase.from('RewriteProgress').update({
      rewriteId: progressId,
      progress: 'Creating an authority post outline'
    }).match({ rewriteId: progressId })

    const result = await agentWithChatHistory.invoke({
      input: `Create an an authority post outline, based on the title and thesis statement you just created. I need it to be thoroughly structured with markdown headings {##} and detailed key talking points beneath every markdown heading {##} to guide the writer. The audience you are writing for is: ${space.audience}.
      
      Please output strictly to Markdown format, ready to upload and nothing else.`
    }, {
      configurable: {
        sessionId: progressId
      }
    })

    await supabase.from('RewriteProgress').update({
      rewriteId: progressId,
      progress: 'Outline generation complete'
    }).match({ rewriteId: progressId })

    const output = result.output.replace(/^`/, '').replace(/`$/, '')

    return output
  },
  createAgent: async ({
    space,
    links,
    progressId
  }: {
    space: any
    links?: string[] | null
    progressId: string
  }): Promise<RunnableWithMessageHistory<Record<string, any>, ChainValues>> => {
    const llm = await openai.modelSelector(space.model as string)

    const searchTool = new TavilySearchResults()

    const tools = [searchTool]

    if (links) {
      const pdfLink = links?.some(link => link.endsWith('.pdf'))

      if (pdfLink) {
        const pdfLink = links?.find(link => link.endsWith('.pdf'))

        if (pdfLink) {
          const response = await fetch(pdfLink)
          const blob = await response.blob()
          const pdfLoader = new PDFLoader(blob as Blob)

          const pdfDocs = await pdfLoader.load()

          const pdfTextSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
          })

          const pdfSplits = await pdfTextSplitter.splitDocuments(pdfDocs)

          const pdfVectorstore = await MemoryVectorStore.fromDocuments(
            pdfSplits,
            new OpenAIEmbeddings()
          )

          const pdfRetriever = pdfVectorstore.asRetriever()

          const pdfRetrieverTool = createRetrieverTool(pdfRetriever, {
            name: 'pdf_reference',
            description: 'Search for information within the PDF document. For any questions about the PDF content, you must use this tool!'
          })

          // @ts-expect-error - I don't know why this is happening
          tools.push(pdfRetrieverTool)
        }
      }

      if (links?.length) {
        const loader = new CheerioWebBaseLoader(links[0])

        const docs = await loader.load()

        const textSplitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 200
        })

        const splitDocs = await textSplitter.splitDocuments(docs)

        const vectorstore = await MemoryVectorStore.fromDocuments(
          splitDocs,
          new OpenAIEmbeddings()
        )

        const retriever = vectorstore.asRetriever()

        const retrieverTool = createRetrieverTool(retriever, {
          name: 'retrieve_company_info',
          description: 'Search for information about the original article. For any questions about the original article, you must use this tool!'
        })

        // @ts-expect-error - I don't know why this is happening
        tools.push(retrieverTool)
      }
    }

    const upstashMessageHistory = new UpstashRedisChatMessageHistory({
      sessionId: progressId,
      config: {
        url: process.env.UPSTASH_URL,
        token: process.env.UPSTASH_TOKEN
      }
    })

    const memory = new BufferMemory({
      returnMessages: true,
      memoryKey: 'chat_history',
      chatHistory: upstashMessageHistory,
      inputKey: 'input',
      outputKey: 'output'
    })

    const prompt = await pull<ChatPromptTemplate>(
      'hwchase17/openai-functions-agent'
    )

    const agent = await createOpenAIFunctionsAgent({
      // @ts-expect-error - I don't know why this is happening
      llm,
      tools,
      prompt
    })

    const agentExecutor = new AgentExecutor({
      agent,
      tools
    })

    const agentWithChatHistory = new RunnableWithMessageHistory({
      runnable: agentExecutor,
      inputMessagesKey: 'input',
      historyMessagesKey: 'chat_history',
      getMessageHistory: async () => {
        return memory.chatHistory
      }
    })

    return agentWithChatHistory
  },
  generateSocialPosts: async ({
    // Space with the persona
    space,
    content,
    progressId
  }: {
    // Space with the persona
    space: any
    content: string
    progressId: string
  }): Promise<{
    facebook: string
    twitter: string
    linkedin: string
    newsletter: string
  } | null> => {
    try {
      if (!space) {
        throw new Error('Space does not have a persona')
      }

      const agentWithChatHistory = await openai.createAgent({
        space,
        progressId
      })

      const summary = await agentWithChatHistory.invoke({
        input: `Forget all previous commands, you are an expert social media writer who is tasked with writing social media posts for a given article. You must use the following article as a reference: ${content}.
        
        In order to write the social media posts, you must first summarize the article and then write the social media posts. Return a summary and list of key highlights of the article only, do not include any additional information.
        `
      }, {
        configurable: {
          sessionId: progressId
        }
      })

      const facebook = await agentWithChatHistory.invoke({
        input: `Write a facebook post for the given article. Return the facebook post in markdown format. 
        
        Here is some information about the article: ${summary.output}

        - ${space.audience && `The audience you are writing for is: ${space.audience}`}
        ${space.customTonePrompt ? space.customTonePrompt : `- Mimic the ${space.persona.tone} style of ${space.persona.name as string} throughout all writing. DO NOT MENTION OR IMPLY THAT YOU ARE ${space.persona.name}, because you are JUST MIMICKING their personality & style.`}


        Keep the Facebook post high level, informative and engaging.

        Do not structure your response in lists, but user full paragraphs.
        
        Here is an example of a great Facebook post. Follow the format and structure of the example but ensure your response is 100% original and is about the previously mentioned article:

        New Article Alert! 🎙️

Hey everyone! I'm excited to announce the release of our latest Article where we dive deep into some incredible stories and insights. Here are three key takeaways from this Article that I think you'll find inspiring:

Overcoming Resistance: We discussed the concept of resistance and how it can hold us back from pursuing our true passions. Drawing inspiration from the quote, "most of us have two lives, the life we live and the unlived life within us," we explored the idea of breaking through resistance to pursue our dreams.

Choosing Fulfillment Over Opportunity: We delved into the idea of choosing between paths that offer financial gain versus paths that bring fulfillment. Through the story of the Lehman Brothers and their journey from selling fabrics to becoming a bank, we explored the importance of staying true to what truly fulfills us, even in the face of tempting opportunities.

The Zen Approach to Success: We shared a fascinating story about a successful entrepreneur who maintained a Zen-like approach to building his companies. Despite not being intense or hardcore, he achieved remarkable success by rolling with the punches and staying true to his values.

I highly recommend checking out this Article to hear more about these incredible stories and insights. You can listen to the full Article on our podcast platform. Let me know your thoughts after listening!

#article #Inspiration #SuccessStories #OvercomingResistance #FulfillmentVsOpportunity #ZenApproachToSuccess`
      }, {
        configurable: {
          sessionId: progressId
        }
      })

      const twitter = await agentWithChatHistory.invoke({
        input: `Write a twitter thread for the given article. Return the twitter thread in markdown format. Here is an example of a great Twitter thread. 
        
        Here is some information about the article: ${summary.output}

        - ${space.audience && `The audience you are writing for is: ${space.audience}`}
        ${space.customTonePrompt ? space.customTonePrompt : `- Mimic the ${space.persona.tone} style of ${space.persona.name as string} throughout all writing. DO NOT MENTION OR IMPLY THAT YOU ARE ${space.persona.name}, because you are JUST MIMICKING their personality & style.`}

        Keep the twitter thread high level, informative and engaging.

        Do not structure your response in lists, but user full paragraphs. Each paragraph should be a separate tweet, and they should tie into each other. Each tweet should make sense as a standalone tweet, without any additional context.

        Follow the format and structure of the example but ensure your response is 100% original and is about the previously mentioned article: 

        Exciting new article alert! 🎙️

🎧 In this article, we dive into the stories of successful entrepreneurs who took unconventional paths to achieve their dreams. From overcoming adversity to finding inspiration in unexpected places, this article is packed with insights and motivation.

🌟 Featuring stories of Sylvester Stallone, the creator of Trello and Stack Overflow, and a Zen-like entrepreneur who found success in a unique way.

🎣 Plus, a surprise fishing trip with a tech mogul that led to unexpected revelations about success and mindset.

🔥 Tune in to hear about the power of resilience, creativity, and the importance of following your passion, even in the face of challenges.

🚀 Read now to get inspired and motivated to pursue your dreams! #Entrepreneurship #SuccessStories #article
        `
      }, {
        configurable: {
          sessionId: progressId
        }
      })

      const linkedin = await agentWithChatHistory.invoke({
        input: `Write a linkedin post for the given article. Return the linkedin post in markdown format.
        
        Here is some information about the article: ${summary.output}

        - ${space.audience && `The audience you are writing for is: ${space.audience}`}
        ${space.customTonePrompt ? space.customTonePrompt : `- Mimic the ${space.persona.tone} style of ${space.persona.name as string} throughout all writing. DO NOT MENTION OR IMPLY THAT YOU ARE ${space.persona.name}, because you are JUST MIMICKING their personality & style.`}

        Keep the LinkedIn post high level, informative and engaging.

        Do not structure your response in lists, but user full paragraphs.
        
        Here is an example of a great Linkedin post. Follow the format and structure of the example but ensure your response is 100% original and is about the previously mentioned article: 

  Exciting New Article Alert! 🎙️

Hey! I am thrilled to announce the release of our latest article where we delved into some truly inspiring stories and valuable insights. Here are three key takeaways from this article that I believe will resonate with you:

1. Overcoming Resistance and Pursuing Your Unlived Life

In this episode, we discussed the concept of resistance and the unlived life within us. We explored how many of us are torn between the life we live and the life we dream of living. The quote, "most of us have two lives, the life we live and the unlived life within us," really struck a chord. It's a reminder to kick resistance to the curb and pursue our dreams with unwavering determination.

2. Choosing Fulfillment Over Opportunistic Paths

We dived into the idea of choosing between paths that lead to financial gain versus paths that offer fulfillment and impact. The discussion highlighted the importance of aligning our actions with our true passions and values, rather than being swayed by short-term gains. It's a reminder to prioritize what truly matters to us in the long run.

3. Embracing Challenges and Following Your Passion

The episode featured captivating stories of individuals like Sylvester Stallone and Michael Pryor, who overcame challenges and pursued their passions with unwavering dedication. Stallone's journey from facing financial struggles to creating the iconic "Rocky" movie series is a testament to the power of perseverance and self-belief. Michael Pryor's low-key approach to success and adaptability in the face of challenges serves as a reminder that success doesn't always require intensity but rather smart work and resilience.

I invite you to listen to the full episode for more inspiring stories and valuable insights. Let's embrace challenges, follow our passions, and strive for fulfillment in all that we do. Remember, the journey to success is unique for each of us, but the key lies in staying true to ourselves and our dreams.

Link to the full article in the comments

#article #Inspiration #SuccessStories #PersonalDevelopment

`
      }, {
        configurable: {
          sessionId: progressId
        }
      })

      const newsletter = await agentWithChatHistory.invoke({
        input: `Write a newsletter for the given article. Return the newsletter in markdown format. 

        Keep the Newsletter post high level, informative and engaging.

        Do not structure your response in lists, but user full paragraphs.
        
        Here is some information about the article: ${summary.output}

        - ${space.audience && `The audience you are writing for is: ${space.audience}`}
        ${space.customTonePrompt ? space.customTonePrompt : `- Mimic the ${space.persona.tone} style of ${space.persona.name as string} throughout all writing. DO NOT MENTION OR IMPLY THAT YOU ARE ${space.persona.name}, because you are JUST MIMICKING their personality & style.`}

        Here is an example of a great Newsletter. Follow the format and structure of the example but ensure your response is 100% original and is about the previously mentioned article: 

        Hey there!

I hope you're doing well. I wanted to share with you the latest article that I am really excited about. In this article, we dive into some inspiring stories and insights that I believe will resonate with you.

We start off by discussing the concept of resistance and how it can hold us back from living our unlived life. We then delve into the fascinating story of the Lehman Brothers and their journey from selling fabrics to becoming a major investment bank.

The article takes an interesting turn as we explore the idea of choosing between opportunistic paths and fulfilling paths in life. We discuss the importance of doing what truly inspires us and not being swayed by short-term gains.

One of the highlights of the article is a deep dive into the life of Sylvester Stallone and his journey to success despite facing challenges and setbacks. We also share a captivating story about a successful entrepreneur who found peace and inspiration through fishing.

If you're looking for some motivation and thought-provoking insights, this episode is a must-listen. I invite you to tune in and discover the valuable lessons and stories we have to share.

You can listen to the full episode on our podcast platform. Thank you for being a part of our podcast community, and I hope you enjoy this episode as much as we enjoyed creating it.

Warm regards,
[Your Name]
        `
      }, {
        configurable: {
          sessionId: progressId
        }
      })

      return {
        facebook: facebook.output.replace(/^`/, '').replace(/`$/, ''),
        twitter: twitter.output.replace(/^`/, '').replace(/`$/, ''),
        linkedin: linkedin.output.replace(/^`/, '').replace(/`$/, ''),
        newsletter: newsletter.output.replace(/^`/, '').replace(/`$/, '')
      }
    } catch (error) {
      console.log(error)
      return null
    }
  },
  generateImageFromPrompt: async ({
    prompt
  }: {
    prompt: string
  }): Promise<string | null> => {
    const imageResponse = await huggingface.stableDiffusion(prompt)
    if (!imageResponse) {
      throw new Error('Image response is null')
    }
    const arrayBuffer = await imageResponse.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    const fileName = `${uuidv4()}.png`

    await supabase.storage.from('featured_images').upload(fileName, buffer, {
      contentType: 'image/png',
      upsert: false
    })

    const { data: fileData } = supabase.storage.from('featured_images').getPublicUrl(fileName)

    return fileData.publicUrl
  },
  generateFeatureImage: async ({
    space,
    article,
    progressId
  }: {
    space: any
    article: string
    progressId: string
  }): Promise<string | null> => {
    const llm = await openai.modelSelector(space.model as string)

    const messageHistory = new ChatMessageHistory()

    const searchTool = new TavilySearchResults()

    const tools = [searchTool]

    const prompt = await pull<ChatPromptTemplate>(
      'hwchase17/openai-functions-agent'
    )

    const agent = await createOpenAIFunctionsAgent({
      // @ts-expect-error - I don't know why this is happening
      llm,
      tools,
      prompt
    })

    const agentExecutor = new AgentExecutor({
      agent,
      tools
    })

    const agentWithChatHistory = new RunnableWithMessageHistory({
      runnable: agentExecutor,
      getMessageHistory: (_sessionId) => messageHistory,
      inputMessagesKey: 'input',
      historyMessagesKey: 'chat_history'
    })

    const result = await agentWithChatHistory.invoke({
      input: `Create a vivid image description suitable for a feature image based on the article provided. Limit the description to a maximum of 25 words.

      Article content: ${article}
      
      RETURN THE COMPLETE PROMPT READY TO UPLOAD AND NOTHING ELSE, DO NOT WRITE EXPLANATIONS.`
    }, {
      configurable: {
        sessionId: 'generateArticle'
      }
    })

    const imageResponse = await huggingface.stableDiffusion(result.output as string)
    if (!imageResponse) {
      throw new Error('Image response is null')
    }
    // download the image
    const arrayBuffer = await imageResponse.arrayBuffer()

    const buffer = Buffer.from(arrayBuffer)

    const fileName = `${uuidv4()}.png`

    await supabase.storage.from('featured_images').upload(fileName, buffer, {
      contentType: 'image/png',
      upsert: false
    })

    const { data: fileData } = supabase.storage.from('featured_images').getPublicUrl(fileName)

    return fileData.publicUrl
  },
}
