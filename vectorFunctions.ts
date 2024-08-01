import { Prisma, type Space, type Stream } from '@prisma/client'
import { prisma } from '../../'

import { PrismaVectorStore } from '@langchain/community/vectorstores/prisma'
import { OpenAIEmbeddings } from '@langchain/openai'
import dayjs from 'dayjs'

const vectorFunctions = {
  cleanStream: (streams, excludedUrls: string[] = ['www.freelancer.com', 'www.youtube.com', 'www.jobs.ac.uk']) => {
    // Filter the response to remove duplicates and exclude certain URLs
    return streams.filter((v, i, a) =>
      a.findIndex((t) => t.metadata.title === v.metadata.title) === i && // Check for duplicate titles
      !excludedUrls.some(url => v.metadata.link.includes(url)) // Exclude specified URLs
    )
  },
  queryStream: async (query: string, limit: number, type: 'STREAM' | 'TRENDING') => {
    const vectorStore = PrismaVectorStore.withModel<Stream>(prisma).create(
      new OpenAIEmbeddings({
        modelName: 'text-embedding-3-large'
      }),
      {
        prisma: Prisma,
        tableName: 'Stream',
        vectorColumnName: 'embeddingLarge',
        columns: {
          link: PrismaVectorStore.IdColumn,
          title: PrismaVectorStore.ContentColumn
        },
        filter: {
          type: {
            equals: type
          },
          date: {
            gte: dayjs().subtract(4, 'days').toDate()
          }
        }
      }
    )

    const streams = await vectorStore.similaritySearch(query, limit)
    return vectorFunctions.cleanStream(streams)
  },
  searchStream: async (space: Space, limit: number, type: 'STREAM' | 'TRENDING') => {
    const vectorStore = PrismaVectorStore.withModel<Stream>(prisma).create(
      new OpenAIEmbeddings({
        modelName: 'text-embedding-3-large'
      }),
      {
        prisma: Prisma,
        tableName: 'Stream',
        vectorColumnName: 'embeddingLarge',
        columns: {
          link: PrismaVectorStore.IdColumn,
          title: PrismaVectorStore.ContentColumn
        },
        filter: {
          type: {
            equals: type
          },
          date: {
            gte: dayjs().subtract(4, 'days').toDate()
          }
        }
      }
    )

    let prompt = ''
    if (space.keywords.length > 0) {
      prompt = space.keywords.join(', ')
    } else {
      prompt = [
        'tech',
        'business',
        'marketing',
        'news'
      ].join(', ')
    }
    const streams = await vectorStore.similaritySearch(prompt, limit)

    return vectorFunctions.cleanStream(streams)
  }
}

export default vectorFunctions
