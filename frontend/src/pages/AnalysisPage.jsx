import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, Search, ChevronDown, ChevronUp, Shield, Zap } from 'lucide-react';
import { getAnomalies, getRootCauses, getModelResults, explainLLM } from '../lib/api';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-all
        ${active ? 'bg-primary text-white shadow-md' : 'text-muted-foreground hover:bg-secondary'}`}
    >
      {children}
    </button>
  );
}

function AnomalyTab({ data }) {
  if (!data) return null;

  const scatterData = data.anomaly_scores.map((score, i) => ({
    index: i,
    score,
    isAnomaly: data.anomaly_indices.includes(i),
  }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center p-4 bg-muted rounded-lg">
          <p className="text-2xl font-bold text-foreground">{data.total_samples}</p>
          <p className="text-xs text-muted-foreground">Total Samples</p>
        </div>
        <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
          <p className="text-2xl font-bold text-orange-600">{data.num_anomalies}</p>
          <p className="text-xs text-muted-foreground">Anomalies</p>
        </div>
        <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
          <p className="text-2xl font-bold text-green-600">{data.num_normal}</p>
          <p className="text-xs text-muted-foreground">Normal</p>
        </div>
      </div>

      {/* Scatter plot */}
      <div className="bg-white rounded-xl border border-border p-5">
        <h3 className="font-semibold text-foreground mb-3">Anomaly Scores Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="index" name="Sample" tick={{ fontSize: 10 }} label={{ value: 'Sample Index', position: 'bottom', fontSize: 11 }} />
            <YAxis dataKey="score" name="Score" tick={{ fontSize: 10 }} label={{ value: 'Anomaly Score', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip formatter={(v) => v.toFixed(4)} contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
            <Scatter data={scatterData.filter(d => !d.isAnomaly)} fill="#22c55e" opacity={0.4} />
            <Scatter data={scatterData.filter(d => d.isAnomaly)} fill="#ef4444" opacity={0.8} />
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 text-xs text-muted-foreground justify-center">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500" /> Normal</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500" /> Anomaly</span>
        </div>
      </div>

      {/* Anomaly details */}
      <div className="bg-white rounded-xl border border-border p-5">
        <h3 className="font-semibold text-foreground mb-3">Anomaly Details (Top {data.anomaly_details.length})</h3>
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {data.anomaly_details.map((det, i) => (
            <div key={i} className="p-3 bg-red-50/50 rounded-lg border border-red-100">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-foreground">
                  Sample #{det.index} {det.failure_flag ? '(FAIL)' : '(PASS)'}
                </span>
                <span className="text-xs px-2 py-0.5 bg-red-100 text-red-700 rounded-full">
                  Score: {det.anomaly_score}
                </span>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {det.top_features.map((f, j) => (
                  <span key={j} className="text-xs px-2 py-1 bg-white rounded border border-border">
                    {f.feature}: <strong>z={f.z_score}</strong>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RootCauseTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  const [explanations, setExplanations] = useState({});
  const [loadingAi, setLoadingAi] = useState(false);

  if (!data || !data.analyses?.length) return <p className="text-muted-foreground">No root cause data available.</p>;

  const handleAskAI = async (analysis, i) => {
    if (explanations[i]) return;
    setLoadingAi(i);
    try {
      const res = await explainLLM({
        sample_index: analysis.sample_index,
        predicted_fail: true,
        root_causes: analysis.root_causes
      });
      setExplanations(prev => ({ ...prev, [i]: res.data.explanation }));
    } catch (e) {
      setExplanations(prev => ({ ...prev, [i]: 'Failed to get explanation.' }));
    }
    setLoadingAi(false);
  };

  return (
    <div className="space-y-3">
      <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200 mb-4">
        <p className="text-2xl font-bold text-red-600">{data.total_failures}</p>
        <p className="text-xs text-muted-foreground">Total Failures Analyzed</p>
      </div>

      {data.analyses.map((analysis, i) => (
        <div key={i} className="bg-white rounded-xl border border-border overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg gradient-danger flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <div className="text-left">
                <p className="text-sm font-semibold text-foreground">Sample #{analysis.sample_index}</p>
                <p className="text-xs text-muted-foreground">{analysis.root_causes.length} root causes · {analysis.anomalies.length} anomalies</p>
              </div>
            </div>
            {expanded === i ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {expanded === i && (
            <div className="px-4 pb-4 space-y-3 border-t border-border pt-3">
              <div className="flex justify-between items-center">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase">Ranked Root Causes</h4>
                <button 
                  onClick={() => handleAskAI(analysis, i)}
                  disabled={loadingAi === i || explanations[i]}
                  className="flex items-center gap-1 text-xs px-3 py-1 bg-primary/10 text-primary rounded hover:bg-primary/20 transition disabled:opacity-50"
                >
                  <Shield className="w-3 h-3" />
                  {loadingAi === i ? 'Thinking...' : explanations[i] ? 'AI Explanation' : 'Ask AI to Explain'}
                </button>
              </div>
              
              {explanations[i] && (
                <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg text-sm text-blue-900 leading-relaxed">
                  <strong>AI Analysis:</strong> {explanations[i]}
                </div>
              )}

              {analysis.root_causes.map((rc, j) => (
                <div key={j} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                  <span className="w-6 h-6 rounded-full gradient-primary flex items-center justify-center text-white text-xs font-bold shrink-0">
                    {rc.rank}
                  </span>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-foreground">{rc.feature}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium
                        ${rc.level === 'CRITICAL' ? 'bg-red-100 text-red-700'
                          : rc.level === 'HIGH' ? 'bg-orange-100 text-orange-700'
                          : 'bg-yellow-100 text-yellow-700'}`}>
                        {rc.level}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Value: {rc.failed_value} → Best pass: {rc.best_pass_value} · z-score: {rc.z_score} · Confidence: {rc.confidence}%
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                      <div className="bg-primary h-1.5 rounded-full" style={{ width: `${rc.confidence}%` }} />
                    </div>
                  </div>
                </div>
              ))}

              {analysis.similar_passing.length > 0 && (
                <>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase mt-4">Similar Passing Samples</h4>
                  {analysis.similar_passing.map((sp, j) => (
                    <div key={j} className="text-xs p-2 bg-green-50 rounded-lg border border-green-100">
                      <span className="font-semibold">Sample #{sp.index}</span> · Similarity: {sp.similarity}
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ShapTab({ data }) {
  if (!data?.shap_summary?.length) return <p className="text-muted-foreground">No SHAP data available.</p>;

  const chartData = data.shap_summary.slice(0, 20).map(s => ({
    name: s.feature.length > 16 ? s.feature.slice(0, 14) + '…' : s.feature,
    fullName: s.feature,
    value: s.mean_abs_shap,
  }));

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-border p-5">
        <h3 className="font-semibold text-foreground mb-3">Mean |SHAP| Values — Top 20 Features</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 10 }} />
            <Tooltip
              formatter={(v, name, props) => [v.toFixed(6), props.payload.fullName]}
              contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
            />
            <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-sample SHAP details */}
      {data.failed_predictions?.length > 0 && (
        <div className="bg-white rounded-xl border border-border p-5">
          <h3 className="font-semibold text-foreground mb-3">SHAP Details for Failed Predictions</h3>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {data.failed_predictions.slice(0, 10).map((pred, i) => (
              <div key={i} className="p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold">
                    Sample #{pred.index}
                    <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${pred.actual === 1 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                      Actual: {pred.actual === 1 ? 'FAIL' : 'PASS'}
                    </span>
                  </span>
                  <span className="text-xs font-mono bg-primary/10 text-primary px-2 py-0.5 rounded">
                    P(fail)={pred.probability}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {pred.shap_details?.map((s, j) => (
                    <span key={j} className={`text-xs px-2 py-1 rounded border ${s.shap_value > 0 ? 'border-red-200 bg-red-50 text-red-700' : 'border-green-200 bg-green-50 text-green-700'}`}>
                      {s.feature}: {s.shap_value > 0 ? '+' : ''}{s.shap_value.toFixed(4)}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalysisPage() {
  const navigate = useNavigate();
  const [tab, setTab] = useState('anomalies');
  const [anomalyData, setAnomalyData] = useState(null);
  const [rcaData, setRcaData] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetch() {
      try {
        const [aRes, rRes, mRes] = await Promise.all([
          getAnomalies(),
          getRootCauses(),
          getModelResults(),
        ]);
        setAnomalyData(aRes.data);
        setRcaData(rRes.data);
        setModelData(mRes.data);
      } catch {
        // not available
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-3 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!anomalyData) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertTriangle className="w-12 h-12 text-warning" />
        <p className="text-muted-foreground font-medium">Run the pipeline first to see analysis results.</p>
        <button onClick={() => navigate('/')} className="px-4 py-2 gradient-primary text-white rounded-lg text-sm font-medium">
          Go to Upload
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-foreground">Deep Analysis</h1>

      {/* Tabs */}
      <div className="flex gap-2 bg-muted p-1 rounded-lg w-fit">
        <TabButton active={tab === 'anomalies'} onClick={() => setTab('anomalies')}>Anomaly Detection</TabButton>
        <TabButton active={tab === 'shap'} onClick={() => setTab('shap')}>SHAP Explanations</TabButton>
        <TabButton active={tab === 'rootcause'} onClick={() => setTab('rootcause')}>Root Causes</TabButton>
      </div>

      {tab === 'anomalies' && <AnomalyTab data={anomalyData} />}
      {tab === 'shap' && <ShapTab data={modelData} />}
      {tab === 'rootcause' && <RootCauseTab data={rcaData} />}
    </div>
  );
}
