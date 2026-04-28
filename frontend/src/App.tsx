import { useState, useCallback, useEffect } from 'react'
import type { AppMode, Feedback, Message } from './types'
import { sendQuery, fetchDocuments, uploadFiles } from './lib/api'
import { playSendSound, playReceiveSound } from './lib/sounds'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import WelcomeScreen from './components/WelcomeScreen'
import ChatArea from './components/ChatArea'
import InputBar from './components/InputBar'

let _id = 0
const uid = () => `msg-${++_id}`

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Sidebar state
  const [mode, setMode] = useState<AppMode>('library')
  const [documents, setDocuments] = useState<string[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [uploadId, setUploadId] = useState<string | undefined>()
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [uploadChunkCount, setUploadChunkCount] = useState(0)
  const [isUploading, setIsUploading] = useState(false)

  // Fetch library documents on mount
  useEffect(() => {
    fetchDocuments()
      .then(setDocuments)
      .catch(() => setDocuments([]))
  }, [])

  const handleSend = useCallback(
    async (query: string) => {
      if (!query.trim() || isLoading) return

      const userMsg: Message = {
        id: uid(),
        role: 'user',
        content: query,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      playSendSound()
      setIsLoading(true)

      try {
        const filterDocs =
          mode === 'library' && selectedDocs.length > 0 && selectedDocs.length < documents.length
            ? selectedDocs
            : undefined

        const activeUploadId = mode === 'upload' ? uploadId : undefined

        const res = await sendQuery(query, filterDocs, activeUploadId)

        const assistantMsg: Message = {
          id: uid(),
          role: 'assistant',
          content: res.answer,
          sources: res.sources,
          response_time: res.response_time,
          from_cache: res.from_cache,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, assistantMsg])
        playReceiveSound()
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: uid(),
            role: 'assistant',
            content:
              "I'm having trouble reaching the server. Please check that the API is running and try again.",
            timestamp: new Date(),
          },
        ])
      } finally {
        setIsLoading(false)
      }
    },
    [isLoading, mode, selectedDocs, documents.length, uploadId],
  )

  const handleUpload = useCallback(async (files: FileList) => {
    setIsUploading(true)
    try {
      const res = await uploadFiles(files, uploadId)
      setUploadId(res.upload_id)
      setUploadedFiles(res.filenames)
      setUploadChunkCount(res.chunk_count)
    } catch {
      alert('Upload failed. Please check the file format and try again.')
    } finally {
      setIsUploading(false)
    }
  }, [uploadId])

  const handleClearUpload = useCallback(() => {
    setUploadId(undefined)
    setUploadedFiles([])
    setUploadChunkCount(0)
    setMessages([])
  }, [])

  const handleFeedback = useCallback(
    (msgId: string, fb: Feedback) => {
      if (fb === 'down') {
        // "Try again" — find the user question before this answer, remove the answer, re-send
        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === msgId)
          if (idx < 1) return prev
          const userMsg = prev[idx - 1]
          if (userMsg.role !== 'user') return prev
          // Remove the assistant message; handleSend will add a new one
          setTimeout(() => handleSend(userMsg.content), 100)
          return prev.filter((m) => m.id !== msgId)
        })
        return
      }
      // "Spot on" or "Useful" — mark feedback and show thanks
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId ? { ...m, feedback: m.feedback === fb ? undefined : fb } : m,
        ),
      )
    },
    [handleSend],
  )

  const handleModeChange = useCallback(
    (m: AppMode) => {
      setMode(m)
      setMessages([])
    },
    [],
  )

  const handleSampleQuestion = useCallback(
    (q: string) => handleSend(q),
    [handleSend],
  )

  const canQuery =
    mode === 'library'
      ? documents.length > 0
      : uploadedFiles.length > 0

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex-1 overflow-hidden flex">
        <Sidebar
          mode={mode}
          onModeChange={handleModeChange}
          documents={documents}
          selectedDocs={selectedDocs}
          onSelectedDocsChange={setSelectedDocs}
          uploadedFiles={uploadedFiles}
          uploadChunkCount={uploadChunkCount}
          isUploading={isUploading}
          onUpload={handleUpload}
          onClearUpload={handleClearUpload}
        />
        <div className="flex-1 flex flex-col overflow-hidden">
          <main className="flex-1 overflow-hidden flex flex-col">
            {messages.length === 0 ? (
              <WelcomeScreen onSampleQuestion={handleSampleQuestion} />
            ) : (
              <ChatArea messages={messages} isLoading={isLoading} onFeedback={handleFeedback} />
            )}
          </main>
          <InputBar onSend={handleSend} isLoading={isLoading} disabled={!canQuery} />
        </div>
      </div>
    </div>
  )
}
