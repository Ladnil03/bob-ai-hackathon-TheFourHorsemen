import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, Send, Loader2, Shield, ShieldAlert, ShieldCheck, ShieldX } from 'lucide-react';
import { predictSample, getOverview } from '../lib/api';

export default function PredictPage() {
  const navigate = useNavigate();
  const [features, setFeatures] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    getOverview().then(() => setReady(true)).catch(() => setReady(false));
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      // Parse the textarea as JSON
      let sample;
      try {
        sample = JSON.parse(JSON.stringify(features));
      } catch {
        sample = features;
      }
      const res = await predictSample(sample);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleTextInput = (text) => {
    try {
      const parsed = JSON.parse(text);
      setFeatures(parsed);
    } catch {
      // ignore parse errors while typing
    }
  };

  if (!ready) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertTriangle className="w-12 h-12 text-warning" />
        <p className="text-muted-foreground font-medium">Run the pipeline first to enable predictions.</p>
        <button onClick={() => navigate('/')} className="px-4 py-2 gradient-primary text-white rounded-lg text-sm font-medium">
          Go to Upload
        </button>
      </div>
    );
  }

  const riskConfig = {
    LOW: { icon: ShieldCheck, color: 'text-green-600', bg: 'bg-green-50 border-green-200', gradient: 'gradient-success' },
    MEDIUM: { icon: Shield, color: 'text-yellow-600', bg: 'bg-yellow-50 border-yellow-200', gradient: 'gradient-warning' },
    HIGH: { icon: ShieldAlert, color: 'text-orange-600', bg: 'bg-orange-50 border-orange-200', gradient: 'gradient-warning' },
    CRITICAL: { icon: ShieldX, color: 'text-red-600', bg: 'bg-red-50 border-red-200', gradient: 'gradient-danger' },
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Predict Failure Risk</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Enter feature values as JSON to score a new sample
        </p>
      </div>

      {/* Input */}
      <div className="bg-white rounded-xl border border-border p-5">
        <label className="text-sm font-medium text-foreground">Sample Features (JSON)</label>
        <textarea
          className="mt-2 w-full h-48 px-4 py-3 font-mono text-sm border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary resize-none"
          placeholder={`{\n  "sensor_0": 3030.93,\n  "sensor_1": 2564.00,\n  "sensor_2": 2187.73\n}`}
          onChange={(e) => handleTextInput(e.target.value)}
        />
        <p className="text-xs text-muted-foreground mt-1">
          Paste a JSON object with feature names matching the selected features. Missing features default to 0.
        </p>

        <button
          onClick={handlePredict}
          disabled={loading}
          className="mt-4 flex items-center gap-2 px-6 py-2.5 gradient-primary text-white rounded-lg font-medium shadow-md hover:shadow-lg transition-all disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Predict
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-50 rounded-xl border border-red-200 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (() => {
        const config = riskConfig[result.classification] || riskConfig.LOW;
        const RiskIcon = config.icon;
        return (
          <div className={`rounded-xl border p-6 ${config.bg} space-y-4`}>
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-xl ${config.gradient} flex items-center justify-center shadow-lg`}>
                <RiskIcon className="w-8 h-8 text-white" />
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Risk Score</p>
                <p className={`text-4xl font-bold ${config.color}`}>{result.risk_score}%</p>
                <p className={`text-sm font-semibold ${config.color}`}>{result.classification} RISK</p>
              </div>
            </div>

            {result.top_features?.length > 0 && (
              <div>
                <p className="text-sm font-semibold text-foreground mb-2">Top Contributing Features</p>
                <div className="space-y-1.5">
                  {result.top_features.map((f, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs font-mono text-muted-foreground w-32 truncate">{f.feature}</span>
                      <div className="flex-1 bg-white/50 rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full"
                          style={{ width: `${Math.min(f.importance * 500, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono">{f.importance.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
