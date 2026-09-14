import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Database, Loader2, CheckCircle2, AlertCircle, FileSpreadsheet, Play } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSecom, uploadCSV, runPipeline, getDatasetInfo } from '../lib/api';

export default function UploadPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [targetCol, setTargetCol] = useState('failure_flag');

  const handleSecom = async () => {
    setLoading(true);
    setError(null);
    const toastId = toast.loading('Loading SECOM dataset...');
    try {
      const res = await useSecom();
      setDatasetInfo(res.data);
      toast.success('SECOM dataset loaded!', { id: toastId });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load SECOM dataset');
      toast.error('Failed to load SECOM', { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      toast.error('Please upload a CSV file');
      return;
    }
    setLoading(true);
    setError(null);
    const toastId = toast.loading('Uploading file...');
    try {
      const res = await uploadCSV(file, targetCol);
      setDatasetInfo(res.data);
      toast.success('File uploaded successfully!', { id: toastId });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload file');
      toast.error('Failed to upload file', { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [targetCol]);

  const handleRunPipeline = async () => {
    setPipelineLoading(true);
    setError(null);
    const toastId = toast.loading('Running ML pipeline & SMOTE augmentation...');
    try {
      await runPipeline(40);
      toast.success('Pipeline finished! View results in Dashboard.', { id: toastId });
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Pipeline failed');
      toast.error('Pipeline failed', { id: toastId });
    } finally {
      setPipelineLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Load Dataset</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Upload your semiconductor manufacturing CSV or use the built-in SECOM dataset
        </p>
      </div>

      {/* Two options side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* SECOM button */}
        <button
          onClick={handleSecom}
          disabled={loading}
          className="card-hover flex flex-col items-center gap-4 p-8 bg-white rounded-xl border-2 border-dashed border-primary/30 hover:border-primary transition-colors cursor-pointer disabled:opacity-50"
        >
          <div className="w-14 h-14 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/25">
            <Database className="w-7 h-7 text-white" />
          </div>
          <div className="text-center">
            <p className="font-semibold text-foreground">Use SECOM Dataset</p>
            <p className="text-xs text-muted-foreground mt-1">
              1,567 samples · 590 sensors · Real semiconductor data
            </p>
          </div>
        </button>

        {/* Upload zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`card-hover flex flex-col items-center gap-4 p-8 bg-white rounded-xl border-2 border-dashed transition-colors cursor-pointer
            ${dragOver ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/50'}`}
        >
          <div className="w-14 h-14 rounded-xl bg-accent/10 flex items-center justify-center">
            <UploadCloud className="w-7 h-7 text-accent" />
          </div>
          <div className="text-center">
            <p className="font-semibold text-foreground">Upload CSV</p>
            <p className="text-xs text-muted-foreground mt-1">
              Drag & drop or click to browse
            </p>
          </div>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => handleFile(e.target.files[0])}
            className="absolute inset-0 opacity-0 cursor-pointer"
            style={{ position: 'relative' }}
          />
        </div>
      </div>

      {/* Target column selector for uploads */}
      <div className="bg-white rounded-xl border border-border p-4">
        <label className="text-sm font-medium text-foreground">Target Column Name</label>
        <input
          type="text"
          value={targetCol}
          onChange={(e) => setTargetCol(e.target.value)}
          placeholder="failure_flag"
          className="mt-1 w-full px-3 py-2 text-sm border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary"
        />
        <p className="text-xs text-muted-foreground mt-1">
          The binary column (0/1 or -1/1) indicating pass/fail
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-3 p-4 bg-blue-50 rounded-xl border border-blue-200">
          <Loader2 className="w-5 h-5 text-primary animate-spin" />
          <span className="text-sm font-medium text-primary">Loading dataset...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-50 rounded-xl border border-red-200">
          <AlertCircle className="w-5 h-5 text-destructive" />
          <span className="text-sm text-destructive">{error}</span>
        </div>
      )}

      {/* Dataset loaded — show summary */}
      {datasetInfo && (
        <div className="bg-white rounded-xl border border-border p-6 space-y-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-success" />
            <h3 className="font-semibold text-foreground">Dataset Loaded</h3>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Samples', value: datasetInfo.summary.total_samples.toLocaleString() },
              { label: 'Features', value: datasetInfo.summary.total_features },
              { label: 'Pass', value: datasetInfo.summary.pass_count.toLocaleString() },
              { label: 'Fail', value: `${datasetInfo.summary.fail_count} (${datasetInfo.summary.failure_rate}%)` },
            ].map(({ label, value }) => (
              <div key={label} className="text-center p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="text-lg font-bold text-foreground">{value}</p>
              </div>
            ))}
          </div>

          {/* Run Pipeline button */}
          <button
            onClick={handleRunPipeline}
            disabled={pipelineLoading}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-primary text-primary-foreground font-semibold rounded-xl shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 transition-all disabled:opacity-50"
          >
            {pipelineLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Running Pipeline (may take 30-60s)...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Run Analysis Pipeline
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
