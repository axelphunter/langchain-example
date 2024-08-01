import { type Space, type Feed, type Rewrites, type Stream } from '@prisma/client'
import { OpenAIEmbeddings } from '@langchain/openai'

export const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-3-small'
})

export const embed = {
  setStream: async (stream: Stream) => {
    const prompt = `${stream.title} ${stream.keywords.join(', ')}`
    const embedding = await embeddings.embedQuery(prompt)

    return embedding
  },
  setRewrite: async (rewrite: Rewrites) => {
    const prompt = `${rewrite.content}`

    const embedding = await embeddings.embedQuery(prompt)

    return embedding
  },
  setFeed: async (feed: Feed) => {
    const prompt = `${feed.title} ${feed.description}`
    const embedding = await embeddings.embedQuery(prompt)

    return embedding
  },
  searchFeed: async (query: string) => {
    const embedding = await embeddings.embedQuery(query)

    return embedding
  },
  matchFeed: async (space: Space) => {
    const prompt = space.keywords.join(', ')

    if (prompt === null) {
      throw new Error('Audience not found')
    }

    const embedding = await embeddings.embedQuery(prompt + ' ' + space.audience)

    return embedding
  },
  matchStream: async (space: Space) => {
    const prompt = space.keywords.join(', ')

    if (prompt === null || prompt === undefined) {
      throw new Error('Audience not found')
    }

    const embedding = await embeddings.embedQuery(prompt + ' ' + space.audience)

    return embedding
  },
  searchStream: async (query: string, space?: Space) => {
    if (space === undefined) {
      throw new Error('Audience not found')
    }

    const prompt = `Search: ${query} Keywords: ${space?.keywords.join(', ')}`
    const embedding = await embeddings.embedQuery(prompt)

    return embedding
  }
}
