import axios from 'axios'
import type { QueryResponse, UploadResponse } from '../types'

const client = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export async function sendQuery(
  query: string,
  filterDocs?: string[],
  uploadId?: string,
): Promise<QueryResponse> {
  const { data } = await client.post<QueryResponse>('/query', {
    query,
    filter_docs: filterDocs?.length ? filterDocs : null,
    upload_id: uploadId ?? null,
  })
  return data
}

export async function fetchDocuments(): Promise<string[]> {
  const { data } = await client.get<{ documents: string[] }>('/documents')
  return data.documents
}

export async function uploadFiles(files: FileList, existingUploadId?: string): Promise<UploadResponse> {
  const form = new FormData()
  Array.from(files).forEach((f) => form.append('files', f))
  if (existingUploadId) form.append('upload_id', existingUploadId)
  const { data } = await client.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return data
}
