export interface Source {
  filename: string
  page: number | string
  score: number
  preview: string
}

export type Feedback = 'up' | 'neutral' | 'down'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  response_time?: number
  from_cache?: boolean
  feedback?: Feedback
  timestamp: Date
}

export interface QueryResponse {
  answer: string
  sources: Source[]
  response_time: number
  retrieval_time: number
  num_chunks: number
  confidence_ok: boolean
  fallback_triggered: boolean
  token_usage: number
  from_cache: boolean
  avg_score: number
  query_type: string
}

export interface UploadResponse {
  upload_id: string
  filenames: string[]
  chunk_count: number
  page_count: number
}

export type AppMode = 'library' | 'upload'
