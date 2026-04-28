import { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import type { Feedback, Message } from '../types'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'

interface ChatAreaProps {
  messages: Message[]
  isLoading: boolean
  onFeedback?: (id: string, fb: Feedback) => void
}

export default function ChatArea({ messages, isLoading, onFeedback }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, isLoading])

  return (
    <div className="flex-1 overflow-y-auto gradient-bg">
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-5">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              isLatest={i === messages.length - 1}
              onFeedback={onFeedback}
            />
          ))}

          {isLoading && (
            <motion.div
              key="typing"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <TypingIndicator />
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
