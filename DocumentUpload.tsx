import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle2, Loader2, FileCode, File } from 'lucide-react';

interface DocumentUploadProps {
  onUploadSuccess: () => void;
  apiEndpoint?: string;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  apiEndpoint = '/api/documents/upload',
}) => {
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const MAX_SIZE_MB = 25;
  const ALLOWED_EXTS = ['.pdf', '.txt', '.md', '.markdown'];

  const validateFile = (file: File): boolean => {
    setErrorMsg(null);
    setSuccessMsg(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      setErrorMsg(`Unsupported file format '${ext}'. Allowed types: PDF, TXT, MD.`);
      return false;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds limit of ${MAX_SIZE_MB} MB.`);
      return false;
    }

    return true;
  };

  const handleFileSelect = (file: File) => {
    if (validateFile(file)) {
      setSelectedFile(file);
    } else {
      setSelectedFile(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const endpoints = Array.from(
      new Set([apiEndpoint, 'http://localhost:8000/api/documents/upload', '/api/documents/upload'])
    );

    let success = false;

    for (const ep of endpoints) {
      try {
        const response = await fetch(ep, {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (response.ok && data.success) {
          setSuccessMsg(`Document '${data.document.filename}' uploaded & processed into ${data.document.chunk_count} vector chunks!`);
          setSelectedFile(null);
          if (fileInputRef.current) fileInputRef.current.value = '';
          onUploadSuccess();
          success = true;
          break;
        } else {
          setErrorMsg(data.error || 'Failed to upload document.');
          success = true;
          break;
        }
      } catch (err) {
        // Try next endpoint
      }
    }

    if (!success) {
      setErrorMsg('Could not connect to API server. Ensure FastAPI backend is running on port 8000.');
    }

    setIsUploading(false);
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return <FileText className="w-8 h-8 text-red-400" />;
    if (ext === 'md' || ext === 'markdown') return <FileCode className="w-8 h-8 text-cyan-400" />;
    return <File className="w-8 h-8 text-emerald-400" />;
  };

  return (
    <div className="bg-[#0F1623] border border-[#1E293B] rounded-2xl p-5 shadow-lg">
      <div className="flex items-center gap-2 mb-3">
        <Upload className="w-5 h-5 text-[#00E5FF]" />
        <h3 className="text-base font-semibold text-white">Upload Knowledge Base Document</h3>
      </div>

      {/* Drag and Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-[#00E5FF] bg-[#00E5FF]/10 scale-[1.01]'
            : 'border-[#1E293B] hover:border-[#00E5FF]/50 bg-[#070A11]/60 hover:bg-[#070A11]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,.markdown"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileSelect(e.target.files[0]);
            }
          }}
        />

        {selectedFile ? (
          <div className="flex flex-col items-center gap-2">
            {getFileIcon(selectedFile.name)}
            <span className="text-sm font-medium text-white max-w-[280px] truncate">{selectedFile.name}</span>
            <span className="text-xs text-slate-400">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to ingest
            </span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <div className="w-12 h-12 rounded-full bg-[#1E293B]/70 flex items-center justify-center text-[#00E5FF]">
              <Upload className="w-6 h-6" />
            </div>
            <p className="text-sm text-slate-300 font-medium">
              <span className="text-[#00E5FF] underline">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-slate-400">PDF, TXT, or Markdown (Max size: 25 MB)</p>
          </div>
        )}
      </div>

      {/* Error Feedback */}
      {errorMsg && (
        <div className="mt-3 p-3 rounded-xl bg-red-950/60 border border-red-500/40 text-red-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Success Feedback */}
      {successMsg && (
        <div className="mt-3 p-3 rounded-xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* Action Buttons */}
      {selectedFile && (
        <div className="mt-4 flex gap-2 justify-end">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setSelectedFile(null);
              setErrorMsg(null);
            }}
            disabled={isUploading}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleUpload();
            }}
            disabled={isUploading}
            className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#00E5FF] hover:bg-[#00B8CC] text-[#070A11] flex items-center gap-2 transition-all shadow-[0_0_15px_-3px_rgba(0,229,255,0.4)] disabled:opacity-50"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Ingesting & Embedding...</span>
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                <span>Start Upload</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
