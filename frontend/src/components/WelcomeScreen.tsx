import { motion } from 'framer-motion'
import { Sparkles, Shield, Zap, FileText } from 'lucide-react'

const features = [
  {
    icon: FileText,
    title: 'Document-Grounded',
    description: 'Every answer comes directly from your documents. Zero hallucinations.',
    iconBg: 'bg-blue-50',
    iconColor: 'text-blue-600',
  },
  {
    icon: Shield,
    title: 'Cited & Verified',
    description: 'Every response shows the exact source page so you can verify instantly.',
    iconBg: 'bg-violet-50',
    iconColor: 'text-violet-600',
  },
  {
    icon: Zap,
    title: 'Instant Answers',
    description: 'Get precise answers in seconds — not hours of manual searching.',
    iconBg: 'bg-amber-50',
    iconColor: 'text-amber-600',
  },
]

const sampleQuestions = [
  'What programming languages does Raj know?',
  'What ML projects has Raj built?',
  'How does the Transformer use attention?',
  'What are the key findings of the GPT-4 report?',
]

interface WelcomeScreenProps {
  onSampleQuestion: (q: string) => void
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

export default function WelcomeScreen({ onSampleQuestion }: WelcomeScreenProps) {
  return (
    <div className="flex-1 overflow-y-auto gradient-bg">
      <motion.div
        className="max-w-2xl mx-auto px-6 py-16"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {/* Hero */}
        <motion.div variants={item} className="text-center mb-12">
          {/* Floating icon with pulse rings */}
          <div className="relative inline-flex items-center justify-center mb-8">
            <span className="absolute w-24 h-24 rounded-full bg-blue-500/10 animate-pulse-ring" />
            <span
              className="absolute w-20 h-20 rounded-full bg-blue-500/10 animate-pulse-ring"
              style={{ animationDelay: '0.6s' }}
            />
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
              className="relative w-20 h-20 rounded-3xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-2xl shadow-blue-200"
            >
              <Sparkles className="w-9 h-9 text-white" />
            </motion.div>
          </div>

          <h1 className="text-4xl font-bold text-gray-900 leading-tight tracking-tight mb-3">
            Ask anything about{' '}
            <span className="gradient-text">your documents</span>
          </h1>
          <p className="text-base text-gray-500 max-w-md mx-auto leading-relaxed">
            Get instant, cited answers from PDFs, Word files, and text documents.
            Built for professionals who need accuracy they can trust.
          </p>
        </motion.div>

        {/* Feature cards */}
        <motion.div variants={item} className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-10">
          {features.map((f) => {
            const Icon = f.icon
            return (
              <motion.div
                key={f.title}
                whileHover={{ y: -4, scale: 1.02 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className="bg-white border border-gray-100 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow cursor-default"
              >
                <div
                  className={`inline-flex p-2.5 rounded-xl ${f.iconBg} mb-3`}
                >
                  <Icon className={`w-4.5 h-4.5 ${f.iconColor}`} size={18} />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1">
                  {f.title}
                </h3>
                <p className="text-xs text-gray-500 leading-relaxed">
                  {f.description}
                </p>
              </motion.div>
            )
          })}
        </motion.div>

        {/* Sample questions */}
        <motion.div variants={item}>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3 text-center">
            Try these questions
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {sampleQuestions.map((q) => (
              <motion.button
                key={q}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => onSampleQuestion(q)}
                className="text-left px-4 py-3 bg-white border border-gray-200 rounded-xl text-sm text-gray-600 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 transition-all duration-150 group shadow-sm"
              >
                <span className="text-blue-400 mr-2 group-hover:text-blue-500 transition-colors">
                  →
                </span>
                {q}
              </motion.button>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </div>
  )
}
