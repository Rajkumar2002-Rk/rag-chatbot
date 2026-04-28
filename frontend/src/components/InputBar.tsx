import { useState, useRef, type KeyboardEvent } from 'react'
import { motion } from 'framer-motion'
import { Send } from 'lucide-react'

interface InputBarProps {
  onSend: (query: string) => void
  isLoading: boolean
  disabled?: boolean
}

export default function InputBar({ onSend, isLoading, disabled }: InputBarProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`
  }

  const canSend = value.trim().length > 0 && !isLoading && !disabled

  return (
    <div className="glass border-t border-gray-100 px-4 py-4">
      <div className="max-w-3xl mx-auto">
        <div
          className={`flex items-end gap-3 bg-white border rounded-2xl px-4 py-3 shadow-sm transition-all duration-200 ${
            canSend
              ? 'border-blue-300 ring-2 ring-blue-50'
              : 'border-gray-200'
          }`}
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder="Ask a question about your documents…"
            disabled={isLoading || disabled}
            rows={1}
            className="flex-1 bg-transparent text-sm text-gray-800 placeholder-gray-400 resize-none outline-none leading-relaxed disabled:opacity-50"
          />
          <motion.button
            whileHover={{ scale: canSend ? 1.05 : 1 }}
            whileTap={{ scale: canSend ? 0.95 : 1 }}
            onClick={handleSend}
            disabled={!canSend}
            className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
              canSend
                ? 'bg-gradient-to-br from-blue-500 to-blue-700 shadow-md shadow-blue-200 cursor-pointer'
                : 'bg-gray-100 cursor-not-allowed'
            }`}
          >
            <Send
              className={`w-3.5 h-3.5 ${canSend ? 'text-white' : 'text-gray-400'}`}
            />
          </motion.button>
        </div>
        <p className="text-center text-xs text-gray-300 mt-2 select-none">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
