import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle, Send, Loader2, Shield, ShieldAlert, ShieldCheck, ShieldX,
  Award, Sliders, Play, Activity, Sparkles, RefreshCw
} from 'lucide-react';
import {
  predictSample,
  getOverview,
  predictBestModel,
  getBestModelSampleWafers,
  getBestModelStatus,
} from '../lib/api';

export default function PredictPage() {
  const navigate = useNavigate();
  const [modelMode, setModelMode] = useState('best'); // 'best' | 'standard'
  const [featuresText, setFeaturesText] = useState('{\n  "sensor_0": 3030.93,\n  "sensor_1": 2564.00,\n  "sensor_2": 2187.73\n}');
  const [features, setFeatures] = useState({ sensor_0: 3030.93, sensor_1: 2564.00, sensor_2: 2187.73 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [standardReady, setStandardReady] = useState(false);
  const [bestReady, setBestReady] = useState(true);
  const [demoWafers, setDemoWafers] = useState([]);
  const [threshold, setThreshold] = useState(0.1518);

  useEffect(() => {
    // Check standard pipeline
    getOverview().then(() => setStandardReady(true)).catch(() => setStandardReady(false));
    // Check best model & get demo wafers
    getBestModelStatus()
      .then(res => {
        setBestReady(res.data?.ready ?? true);
        if (res.data?.overview?.threshold) {
          setThreshold(Number(res.data.overview.threshold));
        }
      })
      .catch(() => setBestReady(false));

    getBestModelSampleWafers()
      .then(res => {
        const samples = res.data?.samples || [];
        setDemoWafers(samples);
        if (samples.length > 0) {
          loadWafer(samples[0]);
        }
      })
      .catch(() => {});
  }, []);

  const loadWafer = (wafer) => {
    const jsonStr = JSON.stringify(wafer.features, null, 2);
    setFeaturesText(jsonStr);
    setFeatures(wafer.features);
    executePrediction(wafer.features, threshold, modelMode);
  };

  const handleTextInput = (text) => {
    setFeaturesText(text);
    try {
      const parsed = JSON.parse(text);
      setFeatures(parsed);
      setError(null);
    } catch {
      // ignore parse errors while typing
    }
  };

  const handlePredictClick = () => {
    let parsed;
    try {
      parsed = JSON.parse(featuresText);
    } catch {
      setError('Invalid JSON format. Please check syntax.');
      return;
    }
    executePrediction(parsed, threshold, modelMode);
  };

  const executePrediction = async (payload, thr, mode) => {
    setLoading(true);
    setError(null);
    try {
      if (mode === 'best') {
        const res = await predictBestModel(payload, thr);
        setResult(res.data);
      } else {
        const res = await predictSample(payload);
        setResult(res.data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const riskConfig = {
    LOW: { icon: ShieldCheck, color: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-200' },
    MEDIUM: { icon: Shield, color: 'text-amber-600', bg: 'bg-amber-50 border-amber-200' },
    HIGH: { icon: ShieldAlert, color: 'text-orange-600', bg: 'bg-orange-50 border-orange-200' },
    CRITICAL: { icon: ShieldX, color: 'text-rose-600', bg: 'bg-rose-50 border-rose-200' },
  };

  const currentRisk = riskConfig[result?.risk_level || 'LOW'] || riskConfig.LOW;
  const RiskIcon = currentRisk.icon;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Wafer Defect Risk Predictor</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Score real or synthetic semiconductor wafers using tuned machine learning models
          </p>
        </div>

        {/* Model Selector Toggle */}
        <div className="inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs font-semibold">
          <button
            onClick={() => {
              setModelMode('best');
              executePrediction(features, threshold, 'best');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all ${
              modelMode === 'best'
                ? 'bg-white text-indigo-900 shadow-sm font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            Best Model (SOTA Blend)
          </button>
          <button
            onClick={() => {
              if (!standardReady) {
                toast.error('Standard pipeline has not been run yet.');
              }
              setModelMode('standard');
              if (standardReady) executePrediction(features, threshold, 'standard');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all ${
              modelMode === 'standard'
                ? 'bg-white text-slate-900 shadow-sm font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Standard Pipeline
          </button>
        </div>
      </div>

      {modelMode === 'standard' && !standardReady ? (
        <div className="flex flex-col items-center justify-center p-12 bg-white rounded-xl border border-border gap-4 text-center">
          <AlertTriangle className="w-12 h-12 text-warning" />
          <div className="space-y-1">
            <h3 className="text-base font-bold text-foreground">Standard Pipeline Not Run</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              The basic pipeline hasn't been executed yet. You can switch to <strong>Best Model (SOTA Blend)</strong> which is already pre-trained and ready immediately, or run the standard pipeline.
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => setModelMode('best')}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700"
            >
              Use Best Model Instead
            </button>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 border border-border text-foreground rounded-lg text-sm font-medium hover:bg-slate-50"
            >
              Run Standard Pipeline
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Input Panel */}
          <div className="lg:col-span-6 space-y-4">
            {/* Quick Demo Buttons */}
            {demoWafers.length > 0 && (
              <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-2">
                <span className="text-xs font-bold text-slate-700 block">1-Click Benchmark Wafers:</span>
                <div className="flex flex-wrap gap-2">
                  {demoWafers.map(w => (
                    <button
                      key={w.id}
                      onClick={() => loadWafer(w)}
                      className="px-2.5 py-1 text-xs rounded-lg border border-slate-200 bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 font-medium text-slate-700 transition-colors"
                    >
                      {w.name.split('—')[1]?.trim() || w.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Threshold Slider (for Best Model) */}
            {modelMode === 'best' && (
              <div className="p-4 rounded-xl bg-white border border-slate-200 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-700 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                    Classification Threshold:
                  </span>
                  <span className="font-mono font-bold text-indigo-600">{threshold.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.50"
                  step="0.005"
                  value={threshold}
                  onChange={e => {
                    const val = parseFloat(e.target.value);
                    setThreshold(val);
                    if (features) executePrediction(features, val, 'best');
                  }}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-medium">
                  <span>0.05 (Catch More Defectives)</span>
                  <span>Optimal: 0.1518</span>
                  <span>0.50 (High Precision)</span>
                </div>
              </div>
            )}

            {/* JSON Input */}
            <div className="bg-white rounded-xl border border-border p-4 space-y-2">
              <label className="text-xs font-bold text-slate-700">Sample Sensor Features (JSON)</label>
              <textarea
                className="w-full h-64 p-3 font-mono text-xs border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary resize-none"
                value={featuresText}
                onChange={e => handleTextInput(e.target.value)}
              />
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-slate-400">
                  {Object.keys(features || {}).length} sensors provided
                </span>
                <button
                  onClick={handlePredictClick}
                  disabled={loading}
                  className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold shadow-sm hover:bg-indigo-700 transition-all disabled:opacity-50"
                >
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                  Evaluate Wafer
                </button>
              </div>
            </div>

            {error && (
              <div className="p-3.5 bg-rose-50 rounded-xl border border-rose-200 text-xs text-rose-700">
                {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-6 space-y-4">
            {loading ? (
              <div className="bg-white rounded-xl border border-border p-16 flex flex-col items-center justify-center gap-3">
                <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                <span className="text-xs font-medium text-muted-foreground">Scoring wafer through {modelMode === 'best' ? '4-way ensemble' : 'baseline'}...</span>
              </div>
            ) : result ? (
              <div className="space-y-4">
                {/* Status Card */}
                <div className="bg-white rounded-xl border border-border p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-border pb-3">
                    <div>
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Classification Outcome</span>
                      <h3 className="text-lg font-bold text-foreground">
                        {result.prediction_label || (result.prediction === 1 ? 'FAIL (Defect Detected)' : 'PASS (Within Spec)')}
                      </h3>
                    </div>
                    <div className={`flex items-center gap-1.5 px-3 py-1 rounded-lg border font-bold text-xs ${currentRisk.bg} ${currentRisk.color}`}>
                      <RiskIcon className="w-4 h-4" />
                      <span>{result.risk_level || (result.prediction === 1 ? 'HIGH' : 'LOW')}</span>
                    </div>
                  </div>

                  {/* Defect Probability Bar */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-slate-600">Defect Probability</span>
                      <span className="font-mono text-slate-900 font-bold">
                        {result.failure_probability_pct ?? ((result.probability ?? 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          (result.predicted_failure ?? result.prediction === 1)
                            ? 'bg-rose-500'
                            : 'bg-emerald-500'
                        }`}
                        style={{
                          width: `${Math.min((result.failure_probability_pct ?? (result.probability * 100)) * 2, 100)}%`
                        }}
                      />
                    </div>
                  </div>

                  {/* Recommendation */}
                  {result.recommended_action && (
                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700">
                      <strong>Fab Action: </strong> {result.recommended_action}
                    </div>
                  )}
                </div>

                {/* Root Causes (Best Model only) */}
                {result.root_causes?.length > 0 && (
                  <div className="bg-white rounded-xl border border-border p-5 space-y-3">
                    <h4 className="text-xs font-bold text-foreground flex items-center gap-1.5">
                      <Activity className="w-4 h-4 text-indigo-600" />
                      Contributing Sensor Deviations (&sigma; Excursion)
                    </h4>
                    <div className="space-y-2">
                      {result.root_causes.slice(0, 5).map((rc, i) => (
                        <div key={i} className="p-2.5 rounded-lg bg-slate-50 border border-slate-100 flex items-center justify-between text-xs">
                          <div>
                            <span className="font-mono font-bold text-slate-800">{rc.feature}</span>
                            <span className="text-[11px] text-slate-400 block">{rc.description}</span>
                          </div>
                          <div className="text-right font-mono">
                            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                              rc.sigma_deviation > 0 ? 'bg-rose-100 text-rose-700' : 'bg-sky-100 text-sky-700'
                            }`}>
                              {rc.sigma_deviation > 0 ? `+${rc.sigma_deviation}σ` : `${rc.sigma_deviation}σ`}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
