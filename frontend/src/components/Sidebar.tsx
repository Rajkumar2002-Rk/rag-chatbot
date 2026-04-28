import { useCallback, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Library,
  Upload,
  FileText,
  Check,
  Trash2,
  Loader2,
  CloudUpload,
} from 'lucide-react'
import type { AppMode } from '../types'

interface SidebarProps {
  mode: AppMode
  onModeChange: (m: AppMode) => void
  documents: string[]
  selectedDocs: string[]
  onSelectedDocsChange: (docs: string[]) => void
  uploadedFiles: string[]
  uploadChunkCount: number
  isUploading: boolean
  onUpload: (files: FileList) => void
  onClearUpload: () => void
}

export default function Sidebar({
  mode,
  onModeChange,
  documents,
  selectedDocs,
  onSelectedDocsChange,
  uploadedFiles,
  uploadChunkCount,
  isUploading,
  onUpload,
  onClearUpload,
}: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const toggleDoc = useCallback(
    (doc: string) => {
      if (selectedDocs.includes(doc)) {
        onSelectedDocsChange(selectedDocs.filter((d) => d !== doc))
      } else {
        onSelectedDocsChange([...selectedDocs, doc])
      }
    },
    [selectedDocs, onSelectedDocsChange],
  )

  const allSelected = selectedDocs.length === documents.length && documents.length > 0

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files.length > 0) {
      onUpload(e.dataTransfer.files)
    }
  }

  return (
    <aside className="w-[260px] flex-shrink-0 bg-white border-r border-gray-100 flex flex-col overflow-hidden">
      {/* Mode toggle */}
      <div className="p-4 border-b border-gray-100">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2.5">
          Mode
        </p>
        <div className="flex gap-1.5 bg-gray-50 p-1 rounded-xl">
          <ModeBtn
            active={mode === 'library'}
            icon={<Library size={14} />}
            label="Library"
            onClick={() => onModeChange('library')}
          />
          <ModeBtn
            active={mode === 'upload'}
            icon={<Upload size={14} />}
            label="Upload"
            onClick={() => onModeChange('upload')}
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <AnimatePresence mode="wait">
          {mode === 'library' ? (
            <motion.div
              key="library"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.15 }}
            >
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
                  Documents
                </p>
                <button
                  onClick={() =>
                    onSelectedDocsChange(allSelected ? [] : [...documents])
                  }
                  className="text-xs text-blue-500 hover:text-blue-600 font-medium"
                >
                  {allSelected ? 'Clear' : 'All'}
                </button>
              </div>

              {documents.length === 0 ? (
                <p className="text-xs text-gray-400 italic">
                  No library found. Run ingest.py first.
                </p>
              ) : (
                <div className="space-y-1">
                  {documents.map((doc) => {
                    const selected = selectedDocs.includes(doc)
                    return (
                      <button
                        key={doc}
                        onClick={() => toggleDoc(doc)}
                        className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-left text-xs transition-all ${
                          selected
                            ? 'bg-blue-50 text-blue-700 border border-blue-200'
                            : 'text-gray-600 hover:bg-gray-50 border border-transparent'
                        }`}
                      >
                        <div
                          className={`w-4 h-4 rounded flex items-center justify-center flex-shrink-0 transition-colors ${
                            selected
                              ? 'bg-blue-500'
                              : 'border border-gray-300'
                          }`}
                        >
                          {selected && <Check size={10} className="text-white" />}
                        </div>
                        <FileText size={13} className="flex-shrink-0 opacity-50" />
                        <span className="truncate">{doc}</span>
                      </button>
                    )
                  })}
                </div>
              )}

              {selectedDocs.length > 0 && selectedDocs.length < documents.length && (
                <p className="mt-3 text-xs text-blue-500 font-medium">
                  Filtering: {selectedDocs.length} of {documents.length} docs
                </p>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="upload"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
            >
              {/* Upload area */}
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragActive(true)
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`flex flex-col items-center justify-center gap-2 p-6 border-2 border-dashed rounded-2xl cursor-pointer transition-all ${
                  dragActive
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                {isUploading ? (
                  <Loader2 size={24} className="text-blue-500 animate-spin" />
                ) : (
                  <CloudUpload
                    size={24}
                    className={dragActive ? 'text-blue-500' : 'text-gray-400'}
                  />
                )}
                <p className="text-xs text-gray-500 text-center">
                  {isUploading
                    ? 'Processing…'
                    : 'Drop files here or click to browse'}
                </p>
                <p className="text-xs text-gray-300">PDF, DOCX, TXT</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      onUpload(e.target.files)
                      e.target.value = ''
                    }
                  }}
                />
              </div>

              {/* Uploaded files list */}
              {uploadedFiles.length > 0 && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
                      Uploaded
                    </p>
                    <button
                      onClick={onClearUpload}
                      className="text-xs text-red-400 hover:text-red-500 flex items-center gap-1"
                    >
                      <Trash2 size={10} />
                      Clear
                    </button>
                  </div>
                  <div className="space-y-1">
                    {uploadedFiles.map((f) => (
                      <div
                        key={f}
                        className="flex items-center gap-2 px-3 py-2 bg-emerald-50 border border-emerald-100 rounded-xl text-xs text-emerald-700"
                      >
                        <FileText size={13} className="flex-shrink-0" />
                        <span className="truncate">{f}</span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-gray-400">
                    {uploadChunkCount} chunks indexed
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </aside>
  )
}

function ModeBtn({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean
  icon: React.ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all ${
        active
          ? 'bg-white text-blue-600 shadow-sm border border-gray-100'
          : 'text-gray-500 hover:text-gray-700'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
