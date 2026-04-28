import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { User, Bot, ChevronDown, Clock, Zap, Crosshair, Lightbulb, RotateCcw } from 'lucide-react'
import Markdown from 'react-markdown'
import type { Feedback, Message } from '../types'
import SourceCard from './SourceCard'
import { useTypewriter } from '../hooks/useTypewriter'

interface MessageBubbleProps {
  message: Message
  isLatest: boolean
  onFeedback?: (id: string, fb: Feedback) => void
}

const reactions: {
  key: Feedback
  icon: typeof Crosshair
  label: string
  color: string
  bg: string
  ring: string
}[] = [
  {
    key: 'up',
    icon: Crosshair,
    label: 'Spot on',
    color: 'text-emerald-600',
    bg: 'bg-emerald-50 hover:bg-emerald-100',
    ring: 'ring-emerald-300',
  },
  {
    key: 'neutral',
    icon: Lightbulb,
    label: 'Useful',
    color: 'text-amber-500',
    bg: 'bg-amber-50 hover:bg-amber-100',
    ring: 'ring-amber-300',
  },
  {
    key: 'down',
    icon: RotateCcw,
    label: 'Try again',
    color: 'text-rose-500',
    bg: 'bg-rose-50 hover:bg-rose-100',
    ring: 'ring-rose-300',
  },
]

export default function MessageBubble({ message, isLatest, onFeedback }: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false)
  const [hoveredKey, setHoveredKey] = useState<Feedback | null>(null)
  const isUser = message.role === 'user'

  const { displayText } = useTypewriter(
    message.role === 'assistant' ? message.content : '',
    isLatest && message.role === 'assistant' ? 18 : 0,
  )

  const content = isUser ? message.content : displayText

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} items-end`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm ${
          isUser
            ? 'bg-gradient-to-br from-blue-500 to-blue-700'
            : 'bg-white border border-gray-200'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {/* Bubble + sources */}
      <div
        className={`flex flex-col gap-2 max-w-[78%] ${
          isUser ? 'items-end' : 'items-start'
        }`}
      >
        {/* Main bubble */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-gradient-to-br from-blue-500 to-blue-700 text-white rounded-br-sm shadow-md shadow-blue-100'
              : 'bg-white border border-gray-100 shadow-sm text-gray-800 rounded-bl-sm'
          }`}
        >
          {content ? (
            isUser ? (
              content
            ) : (
              <Markdown
                components={{
                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                  strong: ({ children }) => (
                    <strong className="font-semibold">{children}</strong>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-4 mb-2 space-y-1">{children}</ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-4 mb-2 space-y-1">{children}</ol>
                  ),
                  li: ({ children }) => <li>{children}</li>,
                  code: ({ children }) => (
                    <code className="bg-gray-100 text-gray-700 px-1 py-0.5 rounded text-xs">
                      {children}
                    </code>
                  ),
                }}
              >
                {content}
              </Markdown>
            )
          ) : (
            <span className="opacity-40 text-xs italic">Thinking...</span>
          )}
        </div>

        {/* Feedback reactions */}
        {!isUser && content && (
          <div className="flex items-center gap-1.5">
            {reactions.map(({ key, icon: Icon, label, color, bg, ring }) => {
              const isSelected = message.feedback === key
              const isOther = message.feedback && !isSelected
              return (
                <motion.button
                  key={key}
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.92 }}
                  onClick={() => onFeedback?.(message.id, key)}
                  onMouseEnter={() => setHoveredKey(key)}
                  onMouseLeave={() => setHoveredKey(null)}
                  className={`relative flex items-center gap-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
                    isSelected
                      ? `${bg} ${color} ring-2 ${ring} px-3 py-1.5 shadow-sm`
                      : isOther
                        ? 'opacity-25 px-2 py-1.5'
                        : `${bg} ${color} px-2 py-1.5 opacity-60 hover:opacity-100 hover:shadow-sm`
                  }`}
                >
                  <motion.span
                    animate={
                      isSelected
                        ? { rotate: [0, -12, 12, -6, 0] }
                        : hoveredKey === key
                          ? { rotate: [0, -8, 8, 0] }
                          : {}
                    }
                    transition={{ duration: 0.4 }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </motion.span>
                  <AnimatePresence>
                    {(isSelected || hoveredKey === key) && (
                      <motion.span
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: 'auto', opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        className="overflow-hidden whitespace-nowrap"
                      >
                        {label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.button>
              )
            })}
            <AnimatePresence>
              {message.feedback && message.feedback !== 'down' && (
                <motion.span
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-xs text-gray-400 ml-1"
                >
                  Thanks for your feedback!
                </motion.span>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setShowSources((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-blue-500 transition-colors mb-1"
            >
              <ChevronDown
                className={`w-3.5 h-3.5 transition-transform duration-200 ${
                  showSources ? 'rotate-180' : ''
                }`}
              />
              {showSources ? 'Hide' : 'Show'} {message.sources.length} source
              {message.sources.length !== 1 ? 's' : ''}
            </button>

            <AnimatePresence>
              {showSources && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-1.5 overflow-hidden"
                >
                  {message.sources.map((src, i) => (
                    <SourceCard key={i} source={src} index={i} />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Meta */}
        {!isUser && (message.response_time || message.from_cache) && (
          <div className="flex items-center gap-2 text-xs text-gray-300">
            {message.from_cache ? (
              <span className="flex items-center gap-1 text-violet-400">
                <Zap className="w-3 h-3" />
                Cached
              </span>
            ) : (
              message.response_time && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {message.response_time.toFixed(1)}s
                </span>
              )
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
