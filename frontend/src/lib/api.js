import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes — pipeline can be slow
});

// Dataset
export const useSecom = () => api.post('/dataset/use-secom');
export const uploadCSV = (file, targetCol = 'failure_flag') => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post(`/upload?target_col=${targetCol}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};
export const getDatasetInfo = () => api.get('/dataset/info');
export const getDatasetPreview = (rows = 10) => api.get(`/dataset/preview?rows=${rows}`);
export const getDatasetColumns = () => api.get('/dataset/columns');

// Pipeline
export const runPipeline = (topK = 50) => api.post(`/pipeline/run?top_k_features=${topK}`);

// Results
export const getOverview = () => api.get('/results/overview');
export const getAnomalies = () => api.get('/results/anomalies');
export const getModelResults = () => api.get('/results/model');
export const getCorrelations = () => api.get('/results/correlations');
export const getRootCauses = () => api.get('/results/root-causes');

export const explainLLM = (payload) => api.post('/results/explain-llm', payload);
export const getFeatureSelection = () => api.get('/results/feature-selection');

// Predict
export const predictSample = (sample) => api.post('/predict', sample);

export default api;
