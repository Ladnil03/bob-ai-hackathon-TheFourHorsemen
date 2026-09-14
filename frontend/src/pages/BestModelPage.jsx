import { useState, useEffect } from 'react';
import {
  Award, Sparkles, Sliders, Play, CheckCircle2, AlertTriangle, XCircle,
  TrendingUp, BarChart3, Layers, Zap, Info, RefreshCw, Cpu, Activity,
  ChevronRight, ArrowUpRight, Gauge, FileText, Bot, Wrench, ShieldCheck, ShieldAlert,
  GitBranch, FlaskConical, Database, ArrowRightLeft, Clock, DollarSign, Target, Beaker
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  CartesianGrid, Legend
} from 'recharts';
import toast from 'react-hot-toast';
import {
  getBestModelMetrics,
  getBestModelSampleWafers,
  predictBestModel,
  prescribeBestModel,
  batchPreflightBestModel,
  getCurvesUrl,
  rlSuggestRecipe,
  rlRecordOutcome,
  rlPolicyDashboard,
  rlSimulateBatch,
  mlflowRegisterCurrent,
  mlflowModelVersions,
  mlflowPromote,
  mlflowExperiments,
} from '../lib/api';

export default function BestModelPage() {
  const [loading, setLoading] = useState(true);
  const [metricsData, setMetricsData] = useState(null);
  const [demoWafers, setDemoWafers] = useState([]);
  const [activeTab, setActiveTab] = useState('predictor');

  // Prediction states
  const [selectedWaferId, setSelectedWaferId] = useState('');
  const [selectedWaferName, setSelectedWaferName] = useState('Wafer-Candidate');
  const [customThreshold, setCustomThreshold] = useState(0.1518);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [activeSamplePayload, setActiveSamplePayload] = useState({});

  // Groq Advisor states
  const [groqAdvice, setGroqAdvice] = useState(null);
  const [groqLoading, setGroqLoading] = useState(false);

  // Batch Pre-Flight states
  const [preflightData, setPreflightData] = useState(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [batchIdInput, setBatchIdInput] = useState('LOT-2026-B09');
  const [chamberInput, setChamberInput] = useState('Chamber-Etch-04B');

  // RL Bandit states
  const [rlPolicy, setRlPolicy] = useState(null);
  const [rlLoading, setRlLoading] = useState(false);
  const [rlSimulating, setRlSimulating] = useState(false);
  const [rlSimResult, setRlSimResult] = useState(null);

  // MLflow states
  const [mlVersions, setMlVersions] = useState(null);
  const [mlExperiments, setMlExperiments] = useState(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlRegistering, setMlRegistering] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [mRes, wRes] = await Promise.all([
        getBestModelMetrics(),
        getBestModelSampleWafers().catch(() => ({ data: { samples: [] } })),
      ]);
      setMetricsData(mRes.data);
      const samples = wRes.data?.samples || [];
      setDemoWafers(samples);

      if (mRes.data?.threshold) {
        setCustomThreshold(Number(mRes.data.threshold));
      }

      // Automatically test the first demo sample (Confirmed Defect)
      if (samples.length > 0) {
        const first = samples[0];
        setSelectedWaferId(first.id);
        setSelectedWaferName(first.name);
        setActiveSamplePayload(first.features);
        runInference(first.features, mRes.data.threshold || 0.1518, first.name);
      }

      runPreflightAnalysis();
    } catch (err) {
      console.error(err);
      toast.error('Failed to load Best Model metrics.');
    } finally {
      setLoading(false);
    }
  };

  const runPreflightAnalysis = async (bId = batchIdInput, chId = chamberInput) => {
    setPreflightLoading(true);
    try {
      const res = await batchPreflightBestModel({ batch_id: bId, chamber_id: chId });
      setPreflightData(res.data);
    } catch (err) {
      console.error('Preflight error:', err);
    } finally {
      setPreflightLoading(false);
    }
  };

  const handleSelectDemoWafer = (sample) => {
    setSelectedWaferId(sample.id);
    setSelectedWaferName(sample.name);
    setActiveSamplePayload(sample.features);
    runInference(sample.features, customThreshold, sample.name);
  };

  const fetchGroqAdvice = async (waferName, predData) => {
    setGroqLoading(true);
    try {
      const pRes = await prescribeBestModel({
        wafer_id: waferName || 'Wafer-Candidate',
        defect_prob_pct: predData.failure_probability_pct,
        threshold: predData.calibrated_threshold,
        risk_level: predData.risk_level,
        root_causes: predData.root_causes || [],
      });
      setGroqAdvice(pRes.data);
    } catch (err) {
      console.error('Groq advice error:', err);
    } finally {
      setGroqLoading(false);
    }
  };

  const runInference = async (features, threshold, name = selectedWaferName) => {
    setPredicting(true);
    try {
      const res = await predictBestModel(features, threshold);
      setPredictionResult(res.data);
      fetchGroqAdvice(name, res.data);
    } catch (err) {
      console.error(err);
      toast.error(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleThresholdChange = (newThr) => {
    setCustomThreshold(newThr);
    if (Object.keys(activeSamplePayload).length > 0) {
      runInference(activeSamplePayload, newThr, selectedWaferName);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-3">
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
        <p className="text-sm font-medium text-muted-foreground">Loading Best Model SOTA Ensemble...</p>
      </div>
    );
  }

  const finalMetrics = metricsData?.final_metrics || {};
  const leaderboard = metricsData?.leaderboard || [];
  const blendWeights = metricsData?.blend_weights || {};

  // Formatted chart data for leaderboard
  const chartData = leaderboard.map(item => ({
    name: item.model.replace(' (WINNER)', ''),
    pr_auc: Number((item.pr_auc * 100).toFixed(1)),
    roc_auc: Number((item.auc_roc * 100).toFixed(1)),
    f1: Number((item.f1 * 100).toFixed(1)),
    accuracy: Number((item.accuracy * 100).toFixed(1)),
    isWinner: item.is_winner,
  }));

  // Weights chart data
  const weightsData = Object.entries(blendWeights).map(([k, v]) => ({
    model: k.toUpperCase(),
    weight: Number((v * 100).toFixed(1)),
  }));

  const riskBadge = {
    LOW: { bg: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20', icon: CheckCircle2 },
    MEDIUM: { bg: 'bg-amber-500/10 text-amber-600 border-amber-500/20', icon: AlertTriangle },
    HIGH: { bg: 'bg-orange-500/10 text-orange-600 border-orange-500/20', icon: AlertTriangle },
    CRITICAL: { bg: 'bg-rose-500/10 text-rose-600 border-rose-500/20', icon: XCircle },
  }[predictionResult?.risk_level || 'LOW'];

  const RiskIcon = riskBadge.icon;

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      {/* Top Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-6 sm:p-8 shadow-xl border border-indigo-900/50">
        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold border border-indigo-400/30">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>IBM Bob AI Hackathon — SOTA Solution</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
              Best Model Ensemble <span className="text-indigo-400">(Winning Blend)</span>
            </h1>
            <p className="text-slate-300 text-sm max-w-2xl">
              Optuna-tuned 4-way ensemble combining <strong>CatBoost</strong>, <strong>LightGBM</strong>, <strong>XGBoost</strong>, and <strong>TabPFN</strong>. Optimized with Differential Evolution blending under strict 5x2 repeated leakage-free cross-validation.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-3">
            <div className="px-4 py-3 rounded-xl bg-white/10 backdrop-blur-md border border-white/10 text-center">
              <span className="text-xs text-slate-300 block font-medium">Ensemble Mode</span>
              <span className="text-lg font-bold text-amber-300 uppercase tracking-wider">
                {metricsData?.ensemble_mode || 'BLEND'}
              </span>
            </div>
            <div className="px-4 py-3 rounded-xl bg-white/10 backdrop-blur-md border border-white/10 text-center">
              <span className="text-xs text-slate-300 block font-medium">Optimal Cutoff</span>
              <span className="text-lg font-bold text-emerald-300 font-mono">
                {metricsData?.threshold ? Number(metricsData.threshold).toFixed(4) : '0.1518'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* PR-AUC */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm relative overflow-hidden group hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">PR-AUC (Avg. Precision)</span>
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-600 flex items-center justify-center font-bold">
              <Award className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900 font-mono">
              {finalMetrics.pr_auc ? Number(finalMetrics.pr_auc).toFixed(4) : '0.2192'}
            </span>
            <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
              +{metricsData?.pr_auc_gain_pct || 77.5}% Lift
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            <strong>{metricsData?.pr_auc_vs_random || '3.3'}×</strong> better than random guess ({metricsData?.random_guess_pr || 0.0664})
          </p>
        </div>

        {/* ROC-AUC */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm relative overflow-hidden group hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">ROC-AUC Score</span>
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-600 flex items-center justify-center font-bold">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900 font-mono">
              {finalMetrics.auc_roc ? Number(finalMetrics.auc_roc).toFixed(4) : '0.7507'}
            </span>
            <span className="text-xs font-semibold text-slate-500">OOF Stratified</span>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            High discrimination power across 1,463 negative wafers
          </p>
        </div>

        {/* F1 Score */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm relative overflow-hidden group hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">F1 Score (Calibrated)</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-600 flex items-center justify-center font-bold">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900 font-mono">
              {finalMetrics.f1 ? Number(finalMetrics.f1).toFixed(4) : '0.3062'}
            </span>
            <span className="text-xs font-medium text-slate-500">
              Thr: {Number(metricsData?.threshold || 0.1518).toFixed(3)}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Balanced Precision ({Number(finalMetrics.precision || 0.305).toFixed(3)}) & Recall ({Number(finalMetrics.recall || 0.308).toFixed(3)})
          </p>
        </div>

        {/* Overall Accuracy */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm relative overflow-hidden group hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Overall Accuracy</span>
            <div className="w-8 h-8 rounded-lg bg-sky-500/10 text-sky-600 flex items-center justify-center font-bold">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900 font-mono">
              {finalMetrics.accuracy ? (Number(finalMetrics.accuracy) * 100).toFixed(1) + '%' : '90.7%'}
            </span>
            <span className="text-xs font-semibold text-slate-500">10-Fold CV</span>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Balanced Error Rate (BER): {Number(finalMetrics.ber || 0.6289).toFixed(4)}
          </p>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-200 pb-2 overflow-x-auto">
        {[
          { id: 'predictor', label: 'Interactive Wafer Predictor', icon: Cpu },
          { id: 'preflight', label: 'Batch Pre-Flight Scorer', icon: ShieldAlert },
          { id: 'rl_optimizer', label: 'RL Recipe Optimizer', icon: FlaskConical },
          { id: 'mlflow', label: 'Model Registry (MLflow)', icon: Database },
          { id: 'leaderboard', label: 'Model Leaderboard & Ensemble', icon: BarChart3 },
          { id: 'curves', label: 'PR & ROC Curves', icon: TrendingUp },
          { id: 'architecture', label: 'Tuning & Hyperparameters', icon: Sliders },
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all whitespace-nowrap ${
                isActive
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab 1: Interactive Predictor */}
      {activeTab === 'predictor' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Sample selection & Threshold */}
          <div className="lg:col-span-5 space-y-6">
            {/* 1-Click Demo Wafers */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                  <Play className="w-4 h-4 text-indigo-600" />
                  1-Click Real Wafer Tests
                </h3>
                <span className="text-xs text-slate-400 font-medium">From SECOM UCI</span>
              </div>
              <p className="text-xs text-slate-500">
                Select a benchmark wafer to immediately test the Best Model without typing 220 sensor numbers:
              </p>

              <div className="space-y-2">
                {demoWafers.map(sample => {
                  const isSelected = selectedWaferId === sample.id;
                  const isDefect = sample.actual_status === 'FAIL';
                  return (
                    <button
                      key={sample.id}
                      onClick={() => handleSelectDemoWafer(sample)}
                      className={`w-full text-left p-3 rounded-lg border transition-all flex items-start justify-between gap-3 ${
                        isSelected
                          ? 'border-indigo-600 bg-indigo-50/50 shadow-sm ring-1 ring-indigo-600'
                          : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/80'
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-xs text-slate-900">{sample.name}</span>
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            isDefect ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
                          }`}>
                            Actual: {sample.actual_status}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500 line-clamp-1">{sample.description}</p>
                      </div>
                      <ChevronRight className={`w-4 h-4 mt-1 transition-transform ${isSelected ? 'text-indigo-600 translate-x-0.5' : 'text-slate-300'}`} />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Threshold Slider */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-indigo-600" />
                  Decision Threshold Slider
                </label>
                <span className="font-mono text-sm font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                  {customThreshold.toFixed(4)}
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.005"
                value={customThreshold}
                onChange={e => handleThresholdChange(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
              <div className="flex justify-between text-[11px] text-slate-400 font-medium">
                <span>0.05 (High Recall / Aggressive)</span>
                <span className="text-indigo-600 font-bold">Optimal: 0.1518</span>
                <span>0.50 (High Precision)</span>
              </div>
              <p className="text-xs text-slate-500 bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                In semiconductor fab lines, missing a defective wafer costs $10k+ downstream. The optimal calibrated threshold (0.1518) maximizes F1 while catching subtle sensor drifts.
              </p>
            </div>
          </div>

          {/* Right Column: Prediction Results & Root Causes */}
          <div className="lg:col-span-7 space-y-6">
            {predicting ? (
              <div className="bg-white rounded-xl border border-slate-200 p-12 flex flex-col items-center justify-center gap-3">
                <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                <span className="text-sm font-medium text-slate-600">Evaluating 4-way blend model...</span>
              </div>
            ) : predictionResult ? (
              <div className="space-y-6">
                {/* Result Card */}
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
                    <div>
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Ensemble Prediction</span>
                      <h2 className="text-xl font-bold text-slate-900 mt-0.5">
                        {predictionResult.prediction_label}
                      </h2>
                    </div>
                    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border font-bold text-xs ${riskBadge.bg}`}>
                      <RiskIcon className="w-4 h-4" />
                      <span>RISK: {predictionResult.risk_level}</span>
                    </div>
                  </div>

                  {/* Probability Gauge Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-slate-600">Defect Probability</span>
                      <span className="font-mono text-slate-900 text-sm font-bold">
                        {predictionResult.failure_probability_pct}%
                      </span>
                    </div>
                    <div className="w-full h-3.5 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          predictionResult.predicted_failure
                            ? 'bg-gradient-to-r from-orange-500 to-rose-600'
                            : 'bg-gradient-to-r from-emerald-400 to-teal-500'
                        }`}
                        style={{ width: `${Math.min(predictionResult.failure_probability_pct * 2.5, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Cutoff Threshold: {(predictionResult.calibrated_threshold * 100).toFixed(1)}%</span>
                      <span>Model Mode: {predictionResult.model_mode.toUpperCase()}</span>
                    </div>
                  </div>

                  {/* Action Recommendation */}
                  <div className="p-3.5 rounded-lg bg-indigo-50/60 border border-indigo-100 text-xs text-indigo-950 flex items-start gap-2.5">
                    <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
                    <div>
                      <strong className="font-semibold block mb-0.5">Recommended Fab Action:</strong>
                      {predictionResult.recommended_action}
                    </div>
                  </div>
                </div>

                {/* Root Cause Attribution */}
                {predictionResult.root_causes?.length > 0 && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                        <Activity className="w-4 h-4 text-indigo-600" />
                        Sensor Drift Root Causes (Attribution)
                      </h3>
                      <span className="text-xs text-slate-400">Top Anomalous Sensors</span>
                    </div>
                    <p className="text-xs text-slate-500">
                      The top sensors exhibiting significant statistical excursion (&sigma;) from the nominal wafer population:
                    </p>

                    <div className="space-y-2.5">
                      {predictionResult.root_causes.map((rc, i) => (
                        <div key={i} className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-xs font-bold text-slate-900">{rc.feature}</span>
                              <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${
                                rc.sigma_deviation > 0 ? 'bg-rose-100 text-rose-700' : 'bg-sky-100 text-sky-700'
                              }`}>
                                {rc.sigma_deviation > 0 ? `+${rc.sigma_deviation}σ` : `${rc.sigma_deviation}σ`}
                              </span>
                            </div>
                            <span className="text-[11px] text-slate-500">{rc.description}</span>
                          </div>

                          <div className="text-right text-xs font-mono">
                            <span className="text-slate-400 text-[10px] block">Val vs Mean</span>
                            <span className="font-bold text-slate-700">{rc.measured_value}</span>
                            <span className="text-slate-400 text-[10px]"> / {rc.reference_mean}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Groq AI Prescriptive Actions */}
                <div className="bg-gradient-to-br from-indigo-900/5 via-slate-50 to-white rounded-xl border border-indigo-200/80 p-6 shadow-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-indigo-100 pb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-sm">
                        <Bot className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="font-bold text-slate-900 text-sm">Groq AI Prescriptive Copilot</h3>
                        <span className="text-[11px] text-slate-500">Root Cause Diagnostics & Equipment Action Plan</span>
                      </div>
                    </div>
                    <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-indigo-100 text-indigo-800 border border-indigo-200 self-start sm:self-auto">
                      {groqAdvice?.powered_by || 'Groq Llama 3.3'}
                    </span>
                  </div>

                  {groqLoading ? (
                    <div className="py-8 flex flex-col items-center justify-center gap-2">
                      <RefreshCw className="w-5 h-5 animate-spin text-indigo-600" />
                      <span className="text-xs text-slate-500 font-medium">Generating prescriptive yield recovery plan...</span>
                    </div>
                  ) : groqAdvice ? (
                    <div className="space-y-4">
                      {/* BLUF */}
                      <div className="p-3.5 rounded-lg bg-indigo-50 border border-indigo-200 text-xs text-indigo-950">
                        <strong className="block font-bold text-indigo-900 mb-1 flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                          Bottom-Line Up-Front (BLUF) Diagnostic:
                        </strong>
                        {groqAdvice.bluf_summary}
                      </div>

                      {/* Mechanism */}
                      <div className="space-y-1 text-xs">
                        <span className="font-bold text-slate-800 block">Underlying Physical Failure Mechanism:</span>
                        <p className="text-slate-600 bg-white p-3 rounded-lg border border-slate-200/80">
                          {groqAdvice.physical_root_cause}
                        </p>
                      </div>

                      {/* Ranked Actions */}
                      {groqAdvice.ranked_corrective_actions?.length > 0 && (
                        <div className="space-y-2">
                          <span className="text-xs font-bold text-slate-800 block">
                            Ranked Corrective Equipment Actions:
                          </span>
                          <div className="space-y-2">
                            {groqAdvice.ranked_corrective_actions.map((act, i) => (
                              <div key={i} className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs flex items-start gap-3">
                                <span className="w-5 h-5 rounded-full bg-slate-900 text-white font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                                  {act.priority || i + 1}
                                </span>
                                <div className="space-y-0.5 text-xs flex-1">
                                  <div className="flex items-center justify-between">
                                    <strong className="font-bold text-slate-900">{act.action}</strong>
                                    <span className="text-[10px] font-semibold text-slate-400 font-mono">
                                      {act.target_subsystem}
                                    </span>
                                  </div>
                                  <p className="text-slate-600 text-[11px]">{act.details}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Bottom Footer: Disposition & Yield Impact */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-indigo-100">
                        <div className="p-2.5 rounded-lg bg-slate-100 border border-slate-200 text-xs">
                          <span className="text-slate-500 text-[10px] block font-medium">Wafer Disposition</span>
                          <strong className="text-slate-900 block font-bold mt-0.5">
                            {groqAdvice.wafer_disposition}
                          </strong>
                        </div>
                        <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs">
                          <span className="text-emerald-700 text-[10px] block font-medium">Projected Yield Recovery</span>
                          <strong className="text-emerald-900 block font-bold mt-0.5">
                            {groqAdvice.projected_yield_recovery_pct}
                          </strong>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                {/* Next Course of Action Roadmap (from GPT-OSS 120B) */}
                {groqAdvice?.next_course_of_action && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
                    <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                      <Target className="w-4 h-4 text-indigo-600" />
                      Next Course of Action — Phased Roadmap
                    </h3>
                    <div className="space-y-3">
                      {[
                        { key: 'immediate_0_to_4h', label: 'Immediate (0–4h)', color: 'rose', icon: '⚡' },
                        { key: 'short_term_4_to_24h', label: 'Short-Term (4–24h)', color: 'amber', icon: '🔧' },
                        { key: 'long_term_1_to_7d', label: 'Long-Term (1–7d)', color: 'emerald', icon: '📋' },
                      ].map(phase => (
                        <div key={phase.key} className={`p-3.5 rounded-lg bg-${phase.color}-50 border border-${phase.color}-200 text-xs`}>
                          <span className={`font-bold text-${phase.color}-900 block mb-1 flex items-center gap-1.5`}>
                            <span>{phase.icon}</span> {phase.label}
                          </span>
                          <p className={`text-${phase.color}-800`}>
                            {groqAdvice.next_course_of_action[phase.key]}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recipe Adjustments Table */}
                {groqAdvice?.recipe_adjustments?.length > 0 && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
                    <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-indigo-600" />
                      Recommended Recipe Parameter Adjustments
                    </h3>
                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                          <tr>
                            <th className="p-3">Parameter</th>
                            <th className="p-3">Current</th>
                            <th className="p-3">Recommended Offset</th>
                            <th className="p-3">Rationale</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {groqAdvice.recipe_adjustments.map((adj, i) => (
                            <tr key={i} className="hover:bg-slate-50/60">
                              <td className="p-3 font-semibold text-slate-900">{adj.parameter}</td>
                              <td className="p-3 font-mono text-slate-500">{adj.current_offset} {adj.unit}</td>
                              <td className="p-3 font-mono font-bold text-indigo-700">
                                {adj.recommended_offset > 0 ? '+' : ''}{adj.recommended_offset} {adj.unit}
                              </td>
                              <td className="p-3 text-slate-600 text-[11px]">{adj.rationale}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Cost Impact Analysis */}
                {groqAdvice?.cost_impact_analysis && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
                    <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                      <DollarSign className="w-4 h-4 text-emerald-600" />
                      Cost Impact Analysis
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs">
                        <span className="text-emerald-700 text-[10px] block font-medium">Act Now Cost</span>
                        <strong className="text-emerald-900 block font-bold text-sm mt-0.5">
                          {groqAdvice.cost_impact_analysis.act_now_cost_usd}
                        </strong>
                      </div>
                      <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs">
                        <span className="text-rose-700 text-[10px] block font-medium">Cost If Deferred</span>
                        <strong className="text-rose-900 block font-bold text-sm mt-0.5">
                          {groqAdvice.cost_impact_analysis.defer_cost_usd}
                        </strong>
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-indigo-50 border border-indigo-200 text-xs text-indigo-900">
                      <strong className="block mb-0.5">ROI Recommendation:</strong>
                      {groqAdvice.cost_impact_analysis.roi_recommendation}
                    </div>
                  </div>
                )}

                {/* SPC Rule Triggers */}
                {groqAdvice?.spc_rule_triggers?.length > 0 && (
                  <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-3">
                    <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-600" />
                      SPC Rule Triggers (Western Electric)
                    </h3>
                    {groqAdvice.spc_rule_triggers.map((rule, i) => (
                      <div key={i} className={`p-3 rounded-lg border text-xs flex items-start gap-2.5 ${
                        rule.severity === 'CRITICAL' ? 'bg-rose-50 border-rose-200 text-rose-900' :
                        rule.severity === 'WARNING' ? 'bg-amber-50 border-amber-200 text-amber-900' :
                        'bg-slate-50 border-slate-200 text-slate-700'
                      }`}>
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                        <div>
                          <strong className="block font-bold">{rule.rule}</strong>
                          <span className="text-[11px]">{rule.description} — Sensor: {rule.sensor}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Tab: Batch Pre-Flight Scorer */}
      {activeTab === 'preflight' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 text-xs font-semibold border border-rose-200 mb-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
                  <span>Problem Statement Challenge #4: Proactive Yield Protection</span>
                </div>
                <h3 className="font-bold text-slate-900 text-lg">Upcoming Batch Pre-Flight Risk Scorer</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Flag upcoming wafer lots whose initial equipment parameters historically correlate with low yield <em>before they run</em>, not after they fail.
                </p>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={batchIdInput}
                  onChange={e => setBatchIdInput(e.target.value)}
                  placeholder="Batch ID"
                  className="px-3 py-1.5 border border-slate-200 rounded-lg text-xs font-mono font-semibold"
                />
                <button
                  onClick={() => runPreflightAnalysis(batchIdInput, chamberInput)}
                  disabled={preflightLoading}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-bold hover:bg-indigo-700 transition-colors flex items-center gap-1.5"
                >
                  {preflightLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  Evaluate Lot
                </button>
              </div>
            </div>

            {preflightLoading ? (
              <div className="py-16 flex flex-col items-center justify-center gap-2">
                <RefreshCw className="w-6 h-6 animate-spin text-indigo-600" />
                <span className="text-xs text-slate-500 font-medium">Scanning lot parameters against historical yield correlations...</span>
              </div>
            ) : preflightData ? (
              <div className="space-y-6">
                {/* Risk Banner */}
                <div className={`p-5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                  preflightData.risk_tier.includes('HIGH') || preflightData.risk_tier.includes('CRITICAL')
                    ? 'bg-rose-50/80 border-rose-200 text-rose-900'
                    : 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
                }`}>
                  <div className="space-y-1">
                    <span className="text-[10px] font-bold uppercase tracking-wider block opacity-70">
                      Pre-Flight Lot Assessment ({preflightData.batch_id} • {preflightData.wafer_count} Wafers)
                    </span>
                    <h4 className="text-xl font-extrabold flex items-center gap-2">
                      <ShieldAlert className="w-5 h-5" />
                      {preflightData.status_flag}
                    </h4>
                    <p className="text-xs opacity-90">{preflightData.recommendation}</p>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <div className="p-3 rounded-lg bg-white/80 border border-current/20 text-center">
                      <span className="text-[10px] block font-medium opacity-75">Projected Lot Yield</span>
                      <span className="text-2xl font-black font-mono">{preflightData.projected_lot_yield_pct}%</span>
                    </div>
                    <div className="p-3 rounded-lg bg-white/80 border border-current/20 text-center">
                      <span className="text-[10px] block font-medium opacity-75">Max Drift (σ)</span>
                      <span className="text-2xl font-black font-mono">+{preflightData.max_parameter_drift_sigma}σ</span>
                    </div>
                  </div>
                </div>

                {/* Flagged Parameters Table */}
                {preflightData.flagged_parameters?.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                        <Activity className="w-4 h-4 text-indigo-600" />
                        Parameters Correlating With Historical Yield Fallout ({preflightData.flagged_sensors_count} Flagged)
                      </h4>
                      <span className="text-[11px] text-slate-400">Comparing Batch Mean vs Golden Baseline</span>
                    </div>

                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                          <tr>
                            <th className="p-3">Sensor Parameter</th>
                            <th className="p-3">Batch Mean</th>
                            <th className="p-3">Golden Baseline</th>
                            <th className="p-3">Sigma Drift (σ)</th>
                            <th className="p-3">Historical Yield Defect Correlation</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 font-mono">
                          {preflightData.flagged_parameters.map((p, idx) => (
                            <tr key={idx} className="hover:bg-slate-50/60">
                              <td className="p-3 font-sans font-semibold text-slate-900">{p.sensor}</td>
                              <td className="p-3 text-slate-700">{p.batch_mean}</td>
                              <td className="p-3 text-slate-500">{p.golden_mean}</td>
                              <td className="p-3 font-bold text-rose-600">+{p.sigma_drift}σ</td>
                              <td className="p-3">
                                <div className="flex items-center gap-2">
                                  <div className="w-20 h-2 bg-slate-200 rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-rose-500 rounded-full"
                                      style={{ width: `${p.correlation_with_low_yield * 100}%` }}
                                    />
                                  </div>
                                  <span className="text-slate-700 font-bold">{(p.correlation_with_low_yield * 100).toFixed(0)}%</span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* Tab 3: Leaderboard */}
      {activeTab === 'leaderboard' && (
        <div className="space-y-6">
          {/* Chart */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-600" />
              Cross-Validation Performance Comparison (5x2 Repeated Stratified CV)
            </h3>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} interval={0} />
                  <YAxis tick={{ fontSize: 11, fill: '#64748b' }} unit="%" domain={[0, 100]} />
                  <Tooltip
                    formatter={(val, name) => [`${val}%`, name.replace('_', ' ').toUpperCase()]}
                    contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e2e8f0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                  <Bar dataKey="pr_auc" name="PR-AUC" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="roc_auc" name="ROC-AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="f1" name="F1 Score" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="accuracy" name="Accuracy" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 flex items-center justify-between">
              <h4 className="font-bold text-slate-900 text-sm">Full CV Metrics Matrix</h4>
              <span className="text-xs text-slate-400">Strict zero-leakage evaluation</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                  <tr>
                    <th className="p-3.5">Model Architecture</th>
                    <th className="p-3.5 font-mono">ROC-AUC</th>
                    <th className="p-3.5 font-mono">PR-AUC</th>
                    <th className="p-3.5 font-mono">F1 Score</th>
                    <th className="p-3.5 font-mono">Precision</th>
                    <th className="p-3.5 font-mono">Recall</th>
                    <th className="p-3.5 font-mono">Accuracy</th>
                    <th className="p-3.5 font-mono">BER</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {leaderboard.map((row, i) => (
                    <tr key={i} className={row.is_winner ? 'bg-indigo-50/70 font-bold text-indigo-950' : 'hover:bg-slate-50/60 text-slate-700'}>
                      <td className="p-3.5 font-sans font-semibold flex items-center gap-2">
                        {row.model}
                        {row.is_winner && (
                          <span className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0.5 rounded font-bold">
                            WINNER
                          </span>
                        )}
                      </td>
                      <td className="p-3.5">{Number(row.auc_roc).toFixed(4)}</td>
                      <td className="p-3.5 text-amber-700 font-bold">{Number(row.pr_auc).toFixed(4)}</td>
                      <td className="p-3.5">{Number(row.f1).toFixed(4)}</td>
                      <td className="p-3.5">{Number(row.precision).toFixed(4)}</td>
                      <td className="p-3.5">{Number(row.recall).toFixed(4)}</td>
                      <td className="p-3.5">{Number(row.accuracy).toFixed(4)}</td>
                      <td className="p-3.5">{Number(row.ber).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Blend Weights Breakdown */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
            <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              Differential Evolution Blend Weights
            </h4>
            <p className="text-xs text-slate-500">
              The optimizer converged on these weights to maximize out-of-fold PR-AUC across the diverse booster models:
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {weightsData.map(w => (
                <div key={w.model} className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 text-center">
                  <span className="text-xs font-semibold text-slate-500 block uppercase">{w.model}</span>
                  <span className="text-2xl font-extrabold text-slate-900 font-mono mt-1 block">
                    {w.weight}%
                  </span>
                  <span className="text-[11px] text-slate-400 mt-0.5 block">Ensemble Weight</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: PR & ROC Curves */}
      {activeTab === 'curves' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
            <div>
              <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-indigo-600" />
                Empirical Evaluation Curves (OOF)
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Precision-Recall and Receiver Operating Characteristic plots generated directly from the repeated 10-fold cross-validation predictions.
              </p>
            </div>
            <a
              href={getCurvesUrl()}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold hover:bg-slate-200 transition-colors"
            >
              Open Full Resolution <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>

          <div className="flex justify-center p-4 bg-slate-50/50 rounded-xl border border-slate-200/60">
            <img
              src={getCurvesUrl()}
              alt="Model Evaluation Curves"
              className="max-h-[500px] w-auto rounded-lg shadow-sm object-contain"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentNode.innerHTML = '<div class="p-8 text-center text-slate-400 text-sm">Curves image available at D:\\bob_models\\r1\\curves.png</div>';
              }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600">
            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
              <strong className="font-bold text-slate-900 block mb-1">Precision-Recall Curve Interpretation</strong>
              With a positive rate of only 6.64%, the PR curve is the primary indicator of reliability. Our blend maintains high precision even at 30-40% recall, meaning flagged wafers have a strong probability of being true yield fallout.
            </div>
            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
              <strong className="font-bold text-slate-900 block mb-1">ROC Curve Interpretation</strong>
              The ROC curve reaches 0.7507 AUC, indicating robust ranking separation between nominal wafers and rare failure events across the entire 590-sensor feature space.
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Tuning & Hyperparameters */}
      {activeTab === 'architecture' && (
        <div className="space-y-6">
          {/* Ablation Summary */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-600" />
              Feature Engineering Forward Ablation
            </h3>
            <p className="text-xs text-slate-500">
              Evaluated 6 candidate feature engineering groups against the baseline top-220 Mutual Information sensors:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                { name: 'A_missing', title: 'Missing Value Indicators', delta: '-0.0073', status: 'Pruned (Overfit)' },
                { name: 'B_rowstats', title: 'Row Statistics (Mean/Std/Skew)', delta: '+0.0001', status: 'Skipped (< +0.005)' },
                { name: 'C_time', title: 'Timestamp Rolling Priors', delta: '-0.0119', status: 'Pruned (Overfit)' },
                { name: 'D_unsupervised', title: 'PCA / KMeans / IsolationForest', delta: '-0.0117', status: 'Pruned (Overfit)' },
                { name: 'E_zscores', title: 'Sensor Z-Scores', delta: '-0.0130', status: 'Pruned (Overfit)' },
                { name: 'E_interactions', title: 'Sensor Interaction Pairs', delta: '-0.0045', status: 'Pruned (Overfit)' },
              ].map(item => (
                <div key={item.name} className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs space-y-1">
                  <div className="flex justify-between font-mono font-bold">
                    <span>{item.name}</span>
                    <span className="text-rose-600">{item.delta}</span>
                  </div>
                  <p className="text-slate-600 text-[11px]">{item.title}</p>
                  <span className="text-[10px] text-slate-400 block font-medium uppercase tracking-wider">{item.status}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 bg-emerald-50 text-emerald-800 p-3 rounded-lg border border-emerald-200">
              <strong>Ablation Conclusion:</strong> The clean top-220 Mutual Information feature subset delivered the highest generalization PR-AUC without noise from synthetic interactions.
            </p>
          </div>

          {/* Hyperparameters */}
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-600" />
              Optuna Best Discovered Hyperparameters (240 Trials)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {['xgb', 'lgbm', 'cat'].map(modelKey => {
                const params = metricsData?.best_params?.[modelKey] || {};
                return (
                  <div key={modelKey} className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                    <span className="font-bold text-slate-900 text-xs uppercase tracking-wider block border-b border-slate-200 pb-1.5">
                      {modelKey.toUpperCase()} Booster
                    </span>
                    <div className="space-y-1.5 font-mono text-[11px]">
                      {Object.entries(params).slice(0, 8).map(([pk, pv]) => (
                        <div key={pk} className="flex justify-between">
                          <span className="text-slate-500">{pk}:</span>
                          <span className="font-bold text-slate-800">
                            {typeof pv === 'number' ? pv.toFixed(4) : String(pv)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Tab: RL Recipe Optimizer */}
      {activeTab === 'rl_optimizer' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-violet-50 text-violet-700 text-xs font-semibold border border-violet-200 mb-1.5">
                  <FlaskConical className="w-3.5 h-3.5 text-violet-600" />
                  <span>Reinforcement Learning — Contextual Bandit</span>
                </div>
                <h3 className="font-bold text-slate-900 text-lg">Continuous Recipe Optimizer</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Thompson Sampling bandit that learns optimal recipe parameter offsets from real-time yield feedback.
                  Each production outcome (PASS/FAIL) updates the posterior, converging on the best recipe for each sensor context.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={async () => {
                    setRlLoading(true);
                    try {
                      const res = await rlPolicyDashboard();
                      setRlPolicy(res.data);
                    } catch(e) { console.error(e); }
                    finally { setRlLoading(false); }
                  }}
                  disabled={rlLoading}
                  className="px-4 py-2 bg-violet-600 text-white rounded-lg text-xs font-bold hover:bg-violet-700 transition-colors flex items-center gap-1.5"
                >
                  {rlLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
                  Load Policy
                </button>
                <button
                  onClick={async () => {
                    setRlSimulating(true);
                    try {
                      const res = await rlSimulateBatch(30);
                      setRlSimResult(res.data);
                      // Refresh policy after simulation
                      const pRes = await rlPolicyDashboard();
                      setRlPolicy(pRes.data);
                    } catch(e) { console.error(e); }
                    finally { setRlSimulating(false); }
                  }}
                  disabled={rlSimulating}
                  className="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-bold hover:bg-slate-900 transition-colors flex items-center gap-1.5"
                >
                  {rlSimulating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Beaker className="w-3.5 h-3.5" />}
                  Simulate 30 Wafers
                </button>
              </div>
            </div>

            {/* Simulation Result */}
            {rlSimResult && (
              <div className="p-4 rounded-xl bg-violet-50 border border-violet-200 text-xs text-violet-900">
                <strong className="block font-bold mb-1">Simulation Complete</strong>
                Processed {rlSimResult.simulated} wafers → {rlSimResult.passes} PASS / {rlSimResult.fails} FAIL ({rlSimResult.yield_rate}% yield).
                Total bandit updates: {rlSimResult.total_updates}.
              </div>
            )}

            {/* Policy Dashboard */}
            {rlPolicy ? (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                    rlPolicy.policy_maturity === 'Converged' ? 'bg-emerald-100 text-emerald-800 border-emerald-300' :
                    rlPolicy.policy_maturity === 'Learning' ? 'bg-amber-100 text-amber-800 border-amber-300' :
                    'bg-slate-100 text-slate-700 border-slate-300'
                  }`}>
                    Policy: {rlPolicy.policy_maturity}
                  </span>
                  <span className="text-xs text-slate-500">
                    {rlPolicy.total_updates} updates across {rlPolicy.n_context_clusters} context clusters
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Algorithm: {rlPolicy.algorithm}
                  </span>
                </div>

                <div className="space-y-3">
                  {rlPolicy.policy?.map(p => (
                    <div key={p.parameter_id} className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="font-bold text-slate-900 text-xs">{p.parameter_name}</span>
                          <span className="text-[10px] text-slate-400 ml-2 font-mono">{p.subsystem}</span>
                        </div>
                        <div className="text-right">
                          <span className="text-xs font-bold text-violet-700 bg-violet-100 px-2 py-0.5 rounded border border-violet-200">
                            Best: {p.best_action_label}
                          </span>
                          <span className="text-[10px] text-slate-400 ml-2">{p.total_observations} obs</span>
                        </div>
                      </div>
                      <div className="flex items-end gap-1 h-16">
                        {p.expected_yields?.map((y, bi) => (
                          <div key={bi} className="flex-1 flex flex-col items-center gap-0.5">
                            <div
                              className={`w-full rounded-t transition-all ${
                                bi === p.best_action_idx ? 'bg-violet-500' : 'bg-slate-300'
                              }`}
                              style={{ height: `${Math.max(y * 100, 5)}%` }}
                            />
                            <span className="text-[9px] text-slate-500 font-mono">{p.bin_labels[bi]}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Recent History */}
                {rlPolicy.recent_history?.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-bold text-slate-900">Recent Outcomes</h4>
                    <div className="flex gap-1.5 flex-wrap">
                      {rlPolicy.recent_history.map((h, i) => (
                        <span key={i} className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                          h.reward > 0.5 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
                        }`}>
                          {h.wafer_id}: {h.reward > 0.5 ? 'PASS' : 'FAIL'}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-xs">
                Click <strong>Load Policy</strong> to view the current RL bandit state, or <strong>Simulate</strong> to generate training data.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab: MLflow Model Registry */}
      {activeTab === 'mlflow' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700 text-xs font-semibold border border-sky-200 mb-1.5">
                  <Database className="w-3.5 h-3.5 text-sky-600" />
                  <span>MLflow Model Lifecycle Management</span>
                </div>
                <h3 className="font-bold text-slate-900 text-lg">Model Registry & Experiment Tracking</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Version, stage-manage, and track experiments for the BestEnsemble model.
                  RL bandit updates are automatically logged as MLflow experiment runs.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={async () => {
                    setMlRegistering(true);
                    try {
                      await mlflowRegisterCurrent();
                      // Refresh
                      const [vRes, eRes] = await Promise.all([mlflowModelVersions(), mlflowExperiments()]);
                      setMlVersions(vRes.data);
                      setMlExperiments(eRes.data);
                      toast.success('Model registered successfully!');
                    } catch(e) { console.error(e); toast.error('Registration failed'); }
                    finally { setMlRegistering(false); }
                  }}
                  disabled={mlRegistering}
                  className="px-4 py-2 bg-sky-600 text-white rounded-lg text-xs font-bold hover:bg-sky-700 transition-colors flex items-center gap-1.5"
                >
                  {mlRegistering ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <GitBranch className="w-3.5 h-3.5" />}
                  Register Current Model
                </button>
                <button
                  onClick={async () => {
                    setMlLoading(true);
                    try {
                      const [vRes, eRes] = await Promise.all([mlflowModelVersions(), mlflowExperiments()]);
                      setMlVersions(vRes.data);
                      setMlExperiments(eRes.data);
                    } catch(e) { console.error(e); }
                    finally { setMlLoading(false); }
                  }}
                  disabled={mlLoading}
                  className="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-bold hover:bg-slate-900 transition-colors flex items-center gap-1.5"
                >
                  {mlLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                  Refresh
                </button>
              </div>
            </div>

            {mlVersions ? (
              <div className="space-y-5">
                {/* Current Production */}
                {mlVersions.current_production && (
                  <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-emerald-900 flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                        Production Model — v{mlVersions.current_production.version}
                      </span>
                      <span className="text-[10px] font-mono text-emerald-600">
                        {mlVersions.current_production.registered_at_iso}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      {['auc_roc', 'pr_auc', 'f1', 'accuracy'].map(mk => (
                        <div key={mk} className="p-2 rounded bg-white border border-emerald-100 text-center">
                          <span className="text-[10px] text-emerald-600 block font-medium uppercase">{mk.replace('_', '-')}</span>
                          <span className="font-bold font-mono text-emerald-900">
                            {Number(mlVersions.current_production.metrics?.[mk] || 0).toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Version Table */}
                {mlVersions.versions?.length > 0 && (
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                        <tr>
                          <th className="p-3">Version</th>
                          <th className="p-3">Stage</th>
                          <th className="p-3 font-mono">PR-AUC</th>
                          <th className="p-3 font-mono">ROC-AUC</th>
                          <th className="p-3 font-mono">F1</th>
                          <th className="p-3">Registered</th>
                          <th className="p-3">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {mlVersions.versions.map(v => (
                          <tr key={v.version} className={v.stage === 'Production' ? 'bg-emerald-50/40' : 'hover:bg-slate-50/60'}>
                            <td className="p-3 font-bold text-slate-900">v{v.version}</td>
                            <td className="p-3">
                              <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                                v.stage === 'Production' ? 'bg-emerald-100 text-emerald-700' :
                                v.stage === 'Staging' ? 'bg-amber-100 text-amber-700' :
                                'bg-slate-100 text-slate-500'
                              }`}>{v.stage}</span>
                            </td>
                            <td className="p-3 font-mono text-amber-700 font-bold">{Number(v.metrics?.pr_auc || 0).toFixed(4)}</td>
                            <td className="p-3 font-mono">{Number(v.metrics?.auc_roc || 0).toFixed(4)}</td>
                            <td className="p-3 font-mono">{Number(v.metrics?.f1 || 0).toFixed(4)}</td>
                            <td className="p-3 text-slate-500 text-[11px]">{v.registered_at_iso}</td>
                            <td className="p-3">
                              {v.stage !== 'Production' && (
                                <button
                                  onClick={async () => {
                                    await mlflowPromote(v.version, 'Production');
                                    const vRes = await mlflowModelVersions();
                                    setMlVersions(vRes.data);
                                    toast.success(`v${v.version} promoted to Production`);
                                  }}
                                  className="text-[10px] font-bold text-sky-700 hover:text-sky-900 underline"
                                >
                                  Promote
                                </button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Experiments Summary */}
                {mlExperiments && (
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                    <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-indigo-600" />
                      RL Experiment Tracker
                    </h4>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200 text-center">
                        <span className="text-[10px] text-slate-500 block">Total Experiments</span>
                        <span className="text-lg font-bold text-slate-900 font-mono">{mlExperiments.total_experiments}</span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200 text-center">
                        <span className="text-[10px] text-slate-500 block">RL Updates</span>
                        <span className="text-lg font-bold text-slate-900 font-mono">{mlExperiments.rl_updates}</span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200 text-center">
                        <span className="text-[10px] text-slate-500 block">Recent Yield Rate</span>
                        <span className="text-lg font-bold text-emerald-700 font-mono">{mlExperiments.recent_yield_rate_pct}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-xs">
                Click <strong>Register Current Model</strong> to add the BestEnsemble to the registry, or <strong>Refresh</strong> to load existing versions.
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
