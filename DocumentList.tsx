import React, { useState } from 'react';
import { FileText, FileCode, File, Trash2, Layers, HardDrive, Clock, AlertTriangle, Loader2 } from 'lucide-react';

export interface DocumentInfo {
  id: string;
  filename: string;

  file_type: string;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  file_hash?: string;
}

interface DocumentListProps {
  documents: DocumentInfo[];
  isLoading: boolean;
  onRefresh: () => void;
  apiEndpoint?: string;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  isLoading,
  onRefresh,
  apiEndpoint = '/api/documents',
}) => {
  const [deleteTarget, setDeleteTarget] = useState<DocumentInfo | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  const getFileIcon = (fileType: string) => {
    const type = fileType.toLowerCase();
    if (type === 'pdf') return <FileText className="w-4 h-4 text-red-400 shrink-0" />;
    if (type === 'md' || type === 'markdown') return <FileCode className="w-4 h-4 text-cyan-400 shrink-0" />;
    return <File className="w-4 h-4 text-emerald-400 shrink-0" />;
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;

    setIsDeleting(true);
    setDeleteError(null);

    const baseEp = apiEndpoint.replace(/\/$/, '');
    const endpoints = Array.from(
      new Set([
        `${baseEp}/${deleteTarget.id}`,
        `http://localhost:8000/api/documents/${deleteTarget.id}`,
        `/api/documents/${deleteTarget.id}`,
      ])
    );

    let success = false;

    for (const ep of endpoints) {
      try {
        const response = await fetch(ep, { method: 'DELETE' });
        if (response.ok) {
          success = true;
          setDeleteTarget(null);
          onRefresh();
          break;
        } else {
          const data = await response.json();
          setDeleteError(data.detail || data.error || 'Failed to delete document.');
          success = true;
          break;
        }
      } catch (err) {
        // Try next endpoint
      }
    }

    if (!success) {
      setDeleteError('Network error while deleting document.');
    }

    setIsDeleting(false);
  };

  const handleClearAll = async () => {
    if (!window.confirm('Wipe ALL vector points and clear all documents from Qdrant Cloud?')) return;

    setIsDeleting(true);
    const endpoints = Array.from(
      new Set([
        apiEndpoint,
        'http://localhost:8000/api/documents',
        '/api/documents',
      ])
    );

    for (const ep of endpoints) {
      try {
        const response = await fetch(ep, { method: 'DELETE' });
        if (response.ok) {
          onRefresh();
          break;
        }
      } catch (err) {
        // Try next
      }
    }
    setIsDeleting(false);
  };

  return (
    <div className="bg-[#0F1623] border border-[#1E293B] rounded-2xl p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-[#00E5FF]" />
          <h3 className="text-base font-semibold text-white">Ingested Documents</h3>
          <span className="text-xs px-2 py-0.5 rounded-full bg-[#1E293B] text-slate-300 font-mono">
            {documents.length}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleClearAll}
            disabled={isLoading || isDeleting}
            className="text-xs px-2.5 py-1 rounded-lg bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 font-medium transition-colors flex items-center gap-1"
            title="Wipe Qdrant Collection & Clear All"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear All Vector Store</span>
          </button>
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="text-xs text-slate-400 hover:text-[#00E5FF] transition-colors p-1.5 rounded-lg hover:bg-[#1E293B]"
            title="Refresh List"
          >
            <Clock className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>


      {/* Document Items List */}
      <div className="flex-1 overflow-y-auto space-y-2.5 max-h-[340px] pr-1">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-10 gap-2 text-slate-400">
            <Loader2 className="w-6 h-6 animate-spin text-[#00E5FF]" />
            <span className="text-xs">Loading documents...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-[#1E293B] rounded-xl text-slate-400 text-xs">
            No documents ingested yet. Upload a PDF, TXT, or Markdown file to populate the vector search store.
          </div>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.id}
              className="p-3.5 rounded-xl bg-[#070A11]/70 border border-[#1E293B] hover:border-[#00E5FF]/40 transition-all flex items-center justify-between gap-3 group"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-9 h-9 rounded-lg bg-[#1E293B]/70 flex items-center justify-center shrink-0">
                  {getFileIcon(doc.file_type)}
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-semibold text-slate-200 truncate group-hover:text-[#00E5FF] transition-colors">
                    {doc.filename}
                  </span>
                  <div className="flex items-center gap-2.5 text-[11px] text-slate-400 mt-0.5">
                    <span className="flex items-center gap-1">
                      <HardDrive className="w-3 h-3 text-slate-500" />
                      {formatSize(doc.size)}
                    </span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Layers className="w-3 h-3 text-slate-500" />
                      {doc.chunk_count} chunks
                    </span>
                    <span>•</span>
                    <span>{formatDate(doc.uploaded_at)}</span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setDeleteTarget(doc)}
                className="w-8 h-8 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 flex items-center justify-center transition-colors shrink-0"
                title="Delete document and vector points"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0F1623] border border-[#1E293B] rounded-2xl max-w-sm w-full p-5 shadow-2xl animate-slide-up">
            <div className="flex items-center gap-3 text-amber-400 mb-3">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-white">Delete Document?</h4>
                <p className="text-xs text-slate-400">This action cannot be undone.</p>
              </div>
            </div>

            <p className="text-xs text-slate-300 mb-4 bg-[#070A11] p-3 rounded-xl border border-[#1E293B] break-all font-mono">
              {deleteTarget.filename} ({deleteTarget.chunk_count} chunks)
            </p>

            {deleteError && (
              <div className="mb-3 p-2.5 rounded-lg bg-red-950/60 border border-red-500/40 text-red-300 text-xs">
                {deleteError}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => {
                  setDeleteTarget(null);
                  setDeleteError(null);
                }}
                disabled={isDeleting}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={isDeleting}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-red-500 hover:bg-red-600 text-white flex items-center gap-2 transition-colors disabled:opacity-50"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Deleting Qdrant points...</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Delete Document</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
