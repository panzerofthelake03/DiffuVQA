"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Plus, Upload, Settings, Moon, Sun, Send, FileText, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "next-themes";

type Message = {
  id: number;
  text: string;
  sender: "user" | "ai";
  isTyping?: boolean;
  fullText?: string;
  fileUrl?: string;
  fileType?: "image" | "pdf";
  fileName?: string;
};

export default function Page() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [hintMessage, setHintMessage] = useState("");
  const [lightboxData, setLightboxData] = useState<{ url: string; type: "image" | "pdf" } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageStartRef = useRef<HTMLDivElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const chunkTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const chunkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const objectUrlsRef = useRef<string[]>([]);
  const hasScrolledStartRef = useRef(false);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Hello, how can I help with your medical query?", sender: "ai" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [messageCount, setMessageCount] = useState(0);
  const [scrollMode, setScrollMode] = useState<"bottom" | "start">("bottom");
  const [activeChunkId, setActiveChunkId] = useState<number | null>(null);
  const [lastImage, setLastImage] = useState<File | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [fontSize, setFontSize] = useState<"sm" | "base" | "lg">("base");
  const [chatWidth, setChatWidth] = useState<"narrow" | "normal" | "wide">("normal");
  const [sendOnEnter, setSendOnEnter] = useState(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    if (selectedFile.type.startsWith("image/") || selectedFile.type === "application/pdf") {
      const objectUrl = URL.createObjectURL(selectedFile);
      objectUrlsRef.current.push(objectUrl);
      setPreviewUrl(objectUrl);
      return;
    }

    setPreviewUrl(null);
  }, [selectedFile]);

  useEffect(() => {
    if (scrollMode === "bottom") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    if (scrollMode === "start" && !hasScrolledStartRef.current && messageStartRef.current) {
      messageStartRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      hasScrolledStartRef.current = true;
    }
  }, [messages, scrollMode, activeChunkId]);

  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
        typingTimeoutRef.current = null;
      }
      if (chunkIntervalRef.current) {
        clearInterval(chunkIntervalRef.current);
        chunkIntervalRef.current = null;
      }
      if (chunkTimeoutRef.current) {
        clearTimeout(chunkTimeoutRef.current);
        chunkTimeoutRef.current = null;
      }
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      objectUrlsRef.current = [];
    };
  }, []);

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (!file) return;
    setSelectedFile(file);
    setHintMessage("");
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setHintMessage("");
  };

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  const clearTimers = () => {
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = null;
    }
    if (chunkTimeoutRef.current) {
      clearTimeout(chunkTimeoutRef.current);
      chunkTimeoutRef.current = null;
    }
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }
  };

  const flushActiveTyping = () => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.isTyping && msg.fullText
          ? { ...msg, text: msg.fullText, isTyping: false, fullText: undefined }
          : msg
      )
    );
    setIsTyping(false);
    clearTimers();
  };

  const handleSendMessage = () => {
    if (!inputValue.trim() && !selectedFile) return;
    if (isTyping) return;

    flushActiveTyping();
    clearTimers();

    const nextCount = messageCount + 1;
    const isFifthMessage = nextCount % 5 === 0;

    // Image only — store it and wait for a question
    if (selectedFile && !inputValue.trim()) {
      const userMsg: Message = {
        id: Date.now(),
        text: "",
        sender: "user",
        fileUrl: previewUrl ?? undefined,
        fileType: selectedFile.type.startsWith("image/") ? "image" : selectedFile.type === "application/pdf" ? "pdf" : undefined,
        fileName: selectedFile.name,
      };
      const aiPrompt: Message = {
        id: Date.now() + 1,
        text: "Image received. What would you like to know about it?",
        sender: "ai",
      };
      setLastImage(selectedFile);
      setMessages((prev) => [...prev, userMsg, aiPrompt]);
      setSelectedFile(null);
      setPreviewUrl(null);
      setHintMessage("");
      setMessageCount(nextCount);
      return;
    }

    // Use current file or fall back to last uploaded image
    const imageToUse = selectedFile ?? lastImage;
    if (selectedFile) setLastImage(selectedFile);

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      sender: "user",
      fileUrl: selectedFile ? (previewUrl ?? undefined) : undefined,
      fileType: selectedFile?.type.startsWith("image/") ? "image" : selectedFile?.type === "application/pdf" ? "pdf" : undefined,
      fileName: selectedFile?.name,
    };

    const typingMessage: Message = {
      id: Date.now() + 1,
      text: "...",
      sender: "ai",
      isTyping: true,
    };

    const fileForFetch = imageToUse;
    const messageToSend = inputValue;

    setMessages((prev) => [...prev, userMessage, typingMessage]);
    setInputValue("");
    setSelectedFile(null);
    setPreviewUrl(null);
    setHintMessage("");
    setIsTyping(true);
    setMessageCount(nextCount);
    setScrollMode(isFifthMessage ? "start" : "bottom");
    hasScrolledStartRef.current = false;

    const buildRequest = (): Promise<Response> => {
      if (fileForFetch) {
        const fd = new FormData();
        fd.append("image", fileForFetch);
        fd.append("question", messageToSend || "What does this medical image show?");
        return fetch("/api/chat", { method: "POST", body: fd });
      }
      return fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageToSend }),
      });
    };

    buildRequest()
      .then(response => response.json())
      .then(data => {
        const aiText = data.text;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === typingMessage.id
              ? { ...msg, text: "", fullText: aiText, isTyping: false }
              : msg
          )
        );
        setActiveChunkId(typingMessage.id);

        chunkTimeoutRef.current = setTimeout(() => {
          if (scrollMode === "start" && messageStartRef.current) {
            messageStartRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
            hasScrolledStartRef.current = true;
          }

          chunkIntervalRef.current = setInterval(() => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === typingMessage.id && msg.fullText) {
                  const nextLength = Math.min(msg.text.length + 50, msg.fullText.length);
                  const nextText = msg.fullText.slice(0, nextLength);

                  if (nextLength >= msg.fullText.length) {
                    if (chunkIntervalRef.current) {
                      clearInterval(chunkIntervalRef.current);
                      chunkIntervalRef.current = null;
                    }
                    setIsTyping(false);
                  }

                  return { ...msg, text: nextText };
                }
                return msg;
              })
            );
          }, 30);
        }, 0);
      })
      .catch(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === typingMessage.id
              ? { ...msg, text: "Sorry, I couldn't reach the medical brain.", isTyping: false }
              : msg
          )
        );
        setIsTyping(false);
      });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && sendOnEnter) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([{ id: 1, text: "Hello, how can I help with your medical query?", sender: "ai" }]);
    setLastImage(null);
    setShowSettings(false);
  };

  const fontSizeClass = fontSize === "sm" ? "text-sm" : fontSize === "lg" ? "text-lg" : "text-base";
  const chatWidthClass = chatWidth === "narrow" ? "max-w-2xl" : chatWidth === "wide" ? "max-w-6xl" : "max-w-4xl";

  // Mock chat history
  const chatHistory = [
    { id: 1, title: "X-ray Analysis Session" },
    { id: 2, title: "Report Summary Chat" },
    { id: 3, title: "General Medical Query" },
  ];

  return (
    <div className="flex h-screen bg-white dark:bg-gray-900">
      {/* Sidebar */}
      <div className="w-64 bg-slate-100 dark:bg-gray-800 border-r border-slate-200 dark:border-gray-700 p-4 flex flex-col">
        <Button variant="outline" className="mb-4 text-slate-900 dark:text-gray-100 border-slate-300 dark:border-gray-600 hover:bg-slate-200 dark:hover:bg-gray-700 dark:bg-gray-800">
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>

        <ScrollArea className="flex-1">
          <div className="space-y-2">
            {chatHistory.map((chat) => (
              <Button
                key={chat.id}
                variant="ghost"
                className="w-full justify-start text-left text-slate-900 dark:text-gray-100 hover:bg-slate-200 dark:hover:bg-gray-700"
              >
                {chat.title}
              </Button>
            ))}
          </div>
        </ScrollArea>

        <Separator className="my-4 dark:bg-gray-700" />

        <div className="flex items-center gap-2">
          <Avatar>
            <AvatarFallback className="text-slate-900 dark:text-gray-100 dark:bg-gray-700">U</AvatarFallback>
          </Avatar>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="text-slate-900 dark:text-gray-100 hover:bg-slate-200 dark:hover:bg-gray-700"
          >
            {mounted && (theme === "light" ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />)}
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setShowSettings(true)} className="text-slate-900 dark:text-gray-100 hover:bg-slate-200 dark:hover:bg-gray-700">
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Main Chat */}
      <div className="flex-1 bg-white dark:bg-gray-900 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4">
          <div className={`${chatWidthClass} mx-auto space-y-4`}>
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                  className={`flex ${
                    message.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    ref={message.id === activeChunkId ? messageStartRef : null}
                    className={`max-w-[80%] break-words whitespace-pre-wrap px-4 py-2 rounded-lg text-slate-900 dark:text-gray-100 ${fontSizeClass} ${
                      message.sender === "user"
                        ? "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
                        : "bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700"
                    }`}
                  >
                    {message.isTyping ? (
                      <div className="flex space-x-1">
                        {[0, 1, 2].map((i) => (
                          <motion.span
                            key={i}
                            className="text-blue-600 text-lg font-bold"
                            animate={{
                              y: [0, -8, 0],
                              opacity: [0.4, 1, 0.4],
                            }}
                            transition={{
                              duration: 0.8,
                              repeat: Infinity,
                              delay: i * 0.2,
                              ease: "easeInOut",
                            }}
                          >
                            •
                          </motion.span>
                        ))}
                      </div>
                    ) : message.fileType === "image" && message.fileUrl ? (
                      <div className="flex flex-col gap-2">
                        <button
                          type="button"
                          onClick={() => setLightboxData({ url: message.fileUrl!, type: "image" })}
                          className="group w-full overflow-hidden rounded-lg border border-slate-200 dark:border-gray-700 bg-slate-50 dark:bg-gray-700 p-0"
                        >
                          <img
                            src={message.fileUrl}
                            alt={message.fileName ?? "Uploaded image"}
                            className="h-auto w-full rounded-lg object-cover transition duration-200 group-hover:scale-105"
                          />
                        </button>
                        {message.text && (
                          <span className="text-sm text-slate-800 dark:text-gray-200">{message.text}</span>
                        )}
                      </div>
                    ) : message.fileType === "pdf" && message.fileUrl ? (
                      <button
                        type="button"
                        onClick={() => setLightboxData({ url: message.fileUrl!, type: "pdf" })}
                        className="w-full rounded-xl border border-slate-200 dark:border-gray-700 bg-slate-50 dark:bg-gray-700 p-3 text-left transition hover:border-slate-300 dark:hover:border-gray-600"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-200 dark:bg-gray-600 text-slate-700 dark:text-gray-200">
                            <FileText className="h-5 w-5" />
                          </div>
                          <div className="min-w-0 text-sm text-slate-900 dark:text-gray-100">
                            <div className="font-medium truncate">{message.fileName}</div>
                            <div className="text-xs text-slate-500 dark:text-gray-400">PDF document</div>
                          </div>
                        </div>
                      </button>
                    ) : (
                      message.text
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="p-4 border-t border-slate-200 dark:border-gray-700">
          <div className="flex flex-col gap-4 mb-4">
            {selectedFile && (
              <div className="flex items-center gap-3 rounded-2xl border border-slate-200 dark:border-gray-700 bg-slate-50 dark:bg-gray-800 px-3 py-2">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt={selectedFile.name}
                    className="h-12 w-12 rounded-lg object-cover"
                  />
                ) : (
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-300">
                    <FileText className="h-5 w-5" />
                  </div>
                )}
                <div className="min-w-0 flex-1 text-sm text-slate-900 dark:text-gray-100">
                  <div className="font-medium truncate">{selectedFile.name}</div>
                  <div className="text-xs text-slate-500 dark:text-gray-400">{selectedFile.type}</div>
                </div>
                <button
                  type="button"
                  onClick={removeSelectedFile}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-500 dark:text-gray-400 transition hover:bg-slate-100 dark:hover:bg-gray-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            {hintMessage && (
              <div className="rounded-2xl border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                {hintMessage}
              </div>
            )}
          </div>

          <div className="relative max-w-4xl mx-auto">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isTyping}
              className="pr-20 rounded-full border border-blue-500 ring-2 ring-blue-500/20 text-slate-900 dark:text-gray-100 placeholder:text-slate-500 dark:placeholder:text-gray-500 bg-white dark:bg-gray-800"
              placeholder="Type your message..."
            />
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-10 top-1/2 -translate-y-1/2 text-slate-900 dark:text-gray-100 hover:bg-slate-200 dark:hover:bg-gray-700"
              onClick={handleFileUpload}
              disabled={isTyping}
            >
              <Upload className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-900 dark:text-gray-100 hover:bg-slate-200 dark:hover:bg-gray-700"
              onClick={handleSendMessage}
              disabled={isTyping || (!inputValue.trim() && !selectedFile)}
            >
              <Send className="w-4 h-4" />
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*,application/pdf"
              className="hidden"
            />
          </div>
        </div>
      </div>

      <AnimatePresence>
        {lightboxData && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-white/95 dark:bg-gray-900/95 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLightboxData(null)}
          >
            <motion.div
              className="relative max-w-[90vw] max-h-[90vh] overflow-hidden rounded-3xl bg-white dark:bg-gray-800 shadow-2xl"
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              onClick={(event) => event.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => setLightboxData(null)}
                className="absolute right-4 top-4 z-10 inline-flex h-10 w-10 items-center justify-center rounded-full bg-slate-100 dark:bg-gray-700 text-slate-900 dark:text-gray-100 shadow-lg"
              >
                <X className="h-5 w-5" />
              </button>
              <div className="h-[80vh] w-[90vw] bg-white dark:bg-gray-800 p-6">
                {lightboxData.type === "pdf" ? (
                  <embed src={lightboxData.url} type="application/pdf" width="100%" height="100%" />
                ) : (
                  <img src={lightboxData.url} alt="Preview" className="h-full w-full object-contain" />
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl w-80 p-6 flex flex-col gap-5"
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-slate-900 dark:text-gray-100">Settings</h2>
                <button onClick={() => setShowSettings(false)} className="text-slate-400 dark:text-gray-500 hover:text-slate-700 dark:hover:text-gray-300">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Theme */}
              <div className="flex flex-col gap-2">
                <span className="text-xs font-medium text-slate-500 dark:text-gray-400 uppercase tracking-wide">Theme</span>
                <div className="flex gap-2">
                  {(["light", "dark"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTheme(t)}
                      className={`flex-1 py-2 rounded-lg border text-sm font-medium capitalize transition ${
                        theme === t
                          ? "bg-blue-500 text-white border-blue-500"
                          : "bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 border-slate-200 dark:border-gray-600 hover:bg-slate-50 dark:hover:bg-gray-600"
                      }`}
                    >
                      {t === "light" ? "☀️ Light" : "🌙 Dark"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Font Size */}
              <div className="flex flex-col gap-2">
                <span className="text-xs font-medium text-slate-500 dark:text-gray-400 uppercase tracking-wide">Font Size</span>
                <div className="flex gap-2">
                  {([["sm", "A", "text-xs"], ["base", "A", "text-sm"], ["lg", "A", "text-base"]] as const).map(([val, label, cls]) => (
                    <button
                      key={val}
                      onClick={() => setFontSize(val)}
                      className={`flex-1 py-2 rounded-lg border font-medium transition ${cls} ${
                        fontSize === val
                          ? "bg-blue-500 text-white border-blue-500"
                          : "bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 border-slate-200 dark:border-gray-600 hover:bg-slate-50 dark:hover:bg-gray-600"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Chat Width */}
              <div className="flex flex-col gap-2">
                <span className="text-xs font-medium text-slate-500 dark:text-gray-400 uppercase tracking-wide">Chat Width</span>
                <div className="flex gap-2">
                  {(["narrow", "normal", "wide"] as const).map((w) => (
                    <button
                      key={w}
                      onClick={() => setChatWidth(w)}
                      className={`flex-1 py-2 rounded-lg border text-sm font-medium capitalize transition ${
                        chatWidth === w
                          ? "bg-blue-500 text-white border-blue-500"
                          : "bg-white dark:bg-gray-700 text-slate-700 dark:text-gray-200 border-slate-200 dark:border-gray-600 hover:bg-slate-50 dark:hover:bg-gray-600"
                      }`}
                    >
                      {w}
                    </button>
                  ))}
                </div>
              </div>

              {/* Send on Enter */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-700 dark:text-gray-300">Send on Enter</span>
                <button
                  onClick={() => setSendOnEnter((v) => !v)}
                  className={`relative w-11 h-6 rounded-full transition-colors ${sendOnEnter ? "bg-blue-500" : "bg-slate-200 dark:bg-gray-600"}`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${sendOnEnter ? "translate-x-5" : "translate-x-0"}`}
                  />
                </button>
              </div>

              {/* Clear Conversation */}
              <button
                onClick={clearChat}
                className="w-full py-2 rounded-lg border border-red-200 text-red-600 text-sm font-medium hover:bg-red-50 transition"
              >
                Clear Conversation
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}