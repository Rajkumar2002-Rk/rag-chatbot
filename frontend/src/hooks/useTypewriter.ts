import { useState, useEffect, useRef } from 'react'

export function useTypewriter(text: string, speed = 18) {
  const [displayText, setDisplayText] = useState('')
  const [isComplete, setIsComplete] = useState(false)
  const prevText = useRef('')

  useEffect(() => {
    // No animation for non-latest or empty
    if (!text || speed === 0) {
      setDisplayText(text)
      setIsComplete(true)
      prevText.current = text
      return
    }

    // Reset when text changes
    setDisplayText('')
    setIsComplete(false)
    prevText.current = text

    const words = text.split(' ')
    let index = 0

    const timer = setInterval(() => {
      if (index < words.length) {
        setDisplayText(prev =>
          index === 0 ? words[0] : prev + ' ' + words[index],
        )
        index++
      } else {
        clearInterval(timer)
        setIsComplete(true)
      }
    }, speed)

    return () => clearInterval(timer)
  }, [text, speed])

  return { displayText, isComplete }
}
