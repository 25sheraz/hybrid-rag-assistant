import React, { useState, useEffect } from 'react';
import { DocumentUpload } from './DocumentUpload';
import { DocumentList, DocumentInfo } from './DocumentList';
import { FolderKanban, RefreshCw } from 'lucide-react';

interface DocumentManagerProps {
  apiBaseUrl?: string;
  className?: string;
}

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  apiBaseUrl = 'http://localhost:8000',
  className = '',
}) => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchDocuments = async () => {
    setIsLoading(true);
    const endpoints = Array.from(
      new Set([
        `${apiBaseUrl}/api/documents`,
        'http://localhost:8000/api/documents',
        '/api/documents',
      ])
    );

    for (const ep of endpoints) {
      try {
        const response = await fetch(ep);
        if (response.ok) {
          const data = await response.json();
          setDocuments(data.documents || []);
          setIsLoading(false);
          return;
        }
      } catch (err) {
        // Try next endpoint
      }
    }

    setIsLoading(false);
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  return (
    <div className={`flex flex-col gap-5 ${className}`}>
      {/* Header Banner */}
      <div className="flex items-center justify-between bg-[#0F1623] border border-[#1E293B] p-4 rounded-2xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#00E5FF] to-blue-600 p-[2px]">
            <div className="w-full h-full bg-[#070A11] rounded-[10px] flex items-center justify-center text-[#00E5FF]">
              <FolderKanban className="w-5 h-5" />
            </div>
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">Document Management System</h2>
            <p className="text-xs text-slate-400">Dynamic document ingestion & Qdrant dual-vector index control</p>
          </div>
        </div>

        <button
          onClick={fetchDocuments}
          disabled={isLoading}
          className="px-3 py-1.5 rounded-xl bg-[#1E293B] hover:bg-slate-700 text-xs font-medium text-slate-200 flex items-center gap-1.5 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <DocumentUpload
          onUploadSuccess={fetchDocuments}
          apiEndpoint={`${apiBaseUrl}/api/documents/upload`}
        />
        <DocumentList
          documents={documents}
          isLoading={isLoading}
          onRefresh={fetchDocuments}
          apiEndpoint={`${apiBaseUrl}/api/documents`}
        />
      </div>
    </div>
  );
};
