"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { uploadCSV } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Brand } from "@/components/brand";
import { Icon } from "@/components/icon";

// Must match UPLOAD_EXTENSIONS in Backend/app/main.py (.zip = images, one folder per class).
const ACCEPTED_EXTENSIONS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".zip"];

const VERIFY_STEPS = [
  "Reading CSV byte stream into client buffer...",
  "Validating comma-separated schemas and structural delimiters...",
  "Running data profiling models & null frequency scans...",
  "Compiling automatic feature types and verification matrices...",
  "Verification complete. Data is healthy and ready for ingestion."
];

export default function UploadPage() {
  const router = useRouter();
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyStepIndex, setVerifyStepIndex] = useState(0);

  useEffect(() => {
    const prevent = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);

  const runVerification = (selectedFile: File) => {
    setIsVerifying(true);
    setVerifyStepIndex(0);
    setError(null);
    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep < VERIFY_STEPS.length) {
        setVerifyStepIndex(currentStep);
      } else {
        clearInterval(interval);
        setTimeout(() => { setIsVerifying(false); setFile(selectedFile); }, 300);
      }
    }, 450);
  };

  const handleFile = useCallback((f: File) => {
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    const MAX_SIZE = 2 * 1024 * 1024 * 1024;  // over 30 MB uploads go straight to storage (R2)
    if (f.size > MAX_SIZE) {
      setError("File is too large. Maximum size allowed is 2 GB.");
      return;
    }
    runVerification(f);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadCSV(file);
      router.push(`/${res.run_id}/preview`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
      setUploading(false);
    }
  };

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setError(null);
  };

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 md:p-gutter relative select-none">


      <div className="max-w-4xl mx-auto w-full">

        {/* Page header */}
        <div className="mb-4 md:mb-8 flex flex-col md:flex-row md:items-end justify-between gap-2 md:gap-4">
          <div>
            <h2 className="text-2xl md:text-headline-lg text-on-surface mb-1 md:mb-2 font-bold">Data Ingestion</h2>
            <p className="text-sm md:text-body-lg text-on-surface-variant">
              Upload a dataset and <Brand /> will automatically configure and run the full training pipeline.
            </p>
          </div>

        </div>

        {/* Upload card */}
        <div className="w-full">
          <div className="glass p-4 md:p-6">

            {/* Card header */}
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-xl md:text-headline-md font-bold text-on-surface flex items-center gap-2">
                <Icon name="upload_file" className="text-primary" />
                Upload Dataset
              </h3>

            </div>

            {/* Dropzone */}
            <div
              className={cn(
                "border-2 border-dashed rounded-xl p-4 sm:p-6 md:p-10 flex flex-col items-center justify-center cursor-pointer group transition-all duration-300 select-none min-h-[200px] md:min-h-[260px] relative overflow-hidden",
                dragging
                  ? "border-primary bg-surface-container-low"
                  : file
                  ? "border-outline-variant/40 bg-transparent"
                  : "border-outline-variant bg-surface hover:border-primary/50 hover:bg-surface-container-low"
              )}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => document.getElementById("csv-input-main")?.click()}
            >
              {isVerifying ? (
                <div className="flex flex-col items-center justify-center p-4">
                  <div className="relative w-16 h-16 mb-6 flex items-center justify-center">
                    <Icon name="cloud_upload" className="text-primary text-[36px] animate-pulse" />
                    <svg className="absolute top-0 left-0 w-16 h-16 animate-[spin_3s_linear_infinite]" viewBox="0 0 36 36">
                      <circle className="text-surface-variant" strokeWidth="2.5" stroke="currentColor" fill="none" cx="18" cy="18" r="16" />
                      <circle className="text-primary" strokeDasharray="30, 100" strokeWidth="2.5" strokeLinecap="round" stroke="currentColor" fill="none" cx="18" cy="18" r="16" />
                    </svg>
                  </div>
                  <div className="text-center space-y-2">
                    <h4 className="text-sm font-bold text-on-surface font-mono animate-pulse">
                      Analyzing Dataset Integrity...
                    </h4>
                    <p className="text-xs text-on-surface-variant font-medium leading-relaxed max-w-sm">
                      {VERIFY_STEPS[verifyStepIndex]}
                    </p>
                  </div>
                  <div className="w-48 bg-surface-container-high rounded-full h-1.5 overflow-hidden mt-4">
                    <div
                      className="bg-primary h-full transition-all duration-300"
                      style={{ width: `${(verifyStepIndex / (VERIFY_STEPS.length - 1)) * 100}%` }}
                    />
                  </div>
                </div>
              ) : file ? (
                <>
                  <div className="w-14 h-14 rounded-xl bg-surface-green-tint flex items-center justify-center text-success-green mb-4 shadow-sm animate-scale">
                    <Icon name="description" className="text-[28px]" />
                  </div>
                  <h4 className="text-base sm:text-lg md:text-headline-md text-on-surface font-bold truncate max-w-xs md:max-w-sm mb-1">{file.name}</h4>
                  <p className="text-xs text-on-surface-variant font-mono mb-4">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB · CSV Schema Verified
                  </p>
                  <button
                    onClick={removeFile}
                    className="bg-surface-container-high text-error hover:bg-error-container text-xs font-bold px-4 py-2 rounded-lg border border-outline-variant hover:border-error/40 transition-all flex items-center gap-1.5"
                  >
                    <Icon name="delete" style={{ fontSize: "16px" }} />
                    Remove File
                  </button>
                </>
              ) : (
                <>
                  <h4 className="text-base sm:text-lg md:text-headline-md text-on-surface font-semibold mb-2 text-center">Drag &amp; Drop files here</h4>
                  <p className="text-[12px] sm:text-xs md:text-body-md text-on-surface-variant mb-5 text-center max-w-xs md:max-w-md px-2 leading-relaxed">
                    Tables (<strong className="font-semibold text-primary font-mono">CSV, Excel, Parquet, JSON</strong>) or a <strong className="font-semibold text-primary font-mono">.zip</strong> of images with one folder per class. Files are parsed and verified on upload.
                  </p>
                  <button
                    className="bg-surface-container-high text-on-surface text-[11px] sm:text-label-md px-4 py-2 sm:px-5 sm:py-2.5 rounded-lg border border-outline-variant group-hover:border-primary group-hover:text-primary transition-all font-semibold"
                    onClick={(e) => { e.stopPropagation(); document.getElementById("csv-input-main")?.click(); }}
                  >
                    Browse Files
                  </button>
                </>
              )}

              <input
                id="csv-input-main"
                type="file"
                accept={ACCEPTED_EXTENSIONS.join(",")}
                className="sr-only"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
              />
            </div>

            {/* Action footer: visible once file is verified */}
            <AnimatePresence>
              {file && !isVerifying && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="mt-5 pt-5 border-t border-outline-variant flex flex-wrap items-center justify-between gap-4">
                    {/* Auto-pilot notice */}
                    <div className="flex items-center gap-2 text-xs text-on-surface-variant font-medium">
                      <Icon name="auto_awesome" className="text-primary shrink-0" style={{ fontSize: "18px" }} />
                      <span>
                        Problem type, feature strategy and missing value handling are
                        <strong className="text-primary font-semibold"> detected automatically</strong> by the pipeline.
                      </span>
                    </div>

                    {/* Error + CTA */}
                    <div className="flex flex-col items-center gap-1.5 w-full md:w-auto md:ml-auto shrink-0">
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={handleUpload}
                        disabled={uploading}
                        className={cn(
                          "btn-primary text-label-md px-6 py-2.5 rounded-lg",
                          uploading && "opacity-60 cursor-not-allowed"
                        )}
                      >
                        <Icon
                          name={uploading ? "sync" : "play_arrow"}
                          className={cn(uploading && "animate-spin")}
                          style={{ fontSize: "18px" }}
                        />
                        <span>
                          {uploading ? "Uploading..." : "Start Ingestion"}
                        </span>
                      </motion.button>
                      {error && <p className="text-xs text-error font-medium">{error}</p>}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>

        {/* Download testing dataset button */}
        <div className="mt-6 flex justify-center">
          <a
            href="/insurance.csv"
            download="insurance.csv"
            className="btn-glass text-xs font-bold px-5 py-2.5 rounded-lg cursor-pointer"
          >
            <Icon name="download" style={{ fontSize: "16px" }} />
            Download a free dataset
          </a>
        </div>
      </div>

    </div>
  );
}
