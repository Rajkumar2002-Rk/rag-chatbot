import { FileText } from 'lucide-react'
import { motion } from 'framer-motion'
import type { Source } from '../types'

interface SourceCardProps {
  source: Source
  index: number
}

export default function SourceCard({ source, index }: SourceCardProps) {
  const matchPercent = Math.min(Math.round(source.score * 200), 100)

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-blue-50/70 border border-blue-100 rounded-xl p-3 hover:bg-blue-50 transition-colors"
    >
      <div className="flex items-start gap-2.5">
        <div className="flex-shrink-0 w-7 h-7 rounded-lg bg-blue-100 flex items-center justify-center mt-0.5">
          <FileText className="w-3.5 h-3.5 text-blue-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <span className="text-xs font-semibold text-blue-900 truncate">
              {source.filename}
            </span>
            <span className="text-xs text-blue-400 flex-shrink-0 font-medium">
              p.{source.page}
            </span>
          </div>
          <p className="text-xs text-blue-700/80 leading-relaxed line-clamp-2 mb-2">
            {source.preview}
          </p>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-blue-200/50 rounded-full h-1">
              <div
                className="bg-blue-500 h-1 rounded-full transition-all"
                style={{ width: `${matchPercent}%` }}
              />
            </div>
            <span className="text-xs text-blue-400 font-medium">{matchPercent}%</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
