import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Layers, Target, TrendingUp, AlertTriangle, CheckCircle, XCircle, ChevronDown } from 'lucide-react';
import { getOverview, getCorrelations, getModelResults } from '../lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Legend,
} from 'recharts';

const COLORS = ['#22c55e', '#ef4444'];
const CHART_COLORS = ['#2563eb', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd'];

function KpiCard({ icon: Icon, label, value, color, sub }) {
  return (
    <div className="card-hover bg-card rounded-xl border border-border p-5 flex items-start gap-4 shadow-sm">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <p className="text-xs text-muted-foreground font-medium">{label}</p>
        <p className="text-xl font-bold text-foreground mt-0.5">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-32 bg-muted rounded-md animate-pulse"></div>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {[1, 2, 3, 4, 5, 6].map(i => (
          <div key={i} className="bg-card rounded-xl border border-border p-5 flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-muted animate-pulse"></div>
            <div className="flex-1 space-y-2">
              <div className="h-3 w-16 bg-muted rounded animate-pulse"></div>
              <div className="h-5 w-24 bg-muted rounded animate-pulse"></div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-card rounded-xl border border-border p-5 h-96 flex flex-col gap-4">
          <div className="h-5 w-48 bg-muted rounded animate-pulse"></div>
          <div className="flex-1 bg-muted/50 rounded animate-pulse"></div>
        </div>
        <div className="bg-card rounded-xl border border-border p-5 h-96 flex flex-col gap-4">
          <div className="h-5 w-40 bg-muted rounded animate-pulse"></div>
          <div className="flex-1 bg-muted/50 rounded animate-pulse rounded-full w-48 h-48 mx-auto mt-8"></div>
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const [overview, setOverview] = useState(null);
  const [correlations, setCorrelations] = useState([]);
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isReportOpen, setIsReportOpen] = useState(false);

  useEffect(() => {
    async function fetch() {
      try {
        const [ovRes, corrRes, modelRes] = await Promise.all([
          getOverview(),
          getCorrelations(),
          getModelResults(),
        ]);
        setOverview(ovRes.data);
        setCorrelations(corrRes.data.correlations?.slice(0, 15) || []);
        setModelData(modelRes.data);
      } catch {
        setOverview(null);
      } finally {
        setLoading(false);
      }
    }
    fetch();
  }, []);

  if (loading) {
    return <DashboardSkeleton />;
  }

  if (!overview) {
    return (
      <div className="flex flex-col items-center justify-center p-12 bg-card rounded-2xl border border-border gap-4 text-center max-w-lg mx-auto mt-12 shadow-sm">
        <div className="w-12 h-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
          <Layers className="w-6 h-6" />
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-bold text-foreground">Standard Pipeline Not Run Yet</h3>
          <p className="text-sm text-muted-foreground">
            The basic analysis pipeline hasn't been triggered, but the <strong>Optuna-Tuned Best Model (SOTA Ensemble)</strong> is already pre-trained and ready to inspect with full cross-validation metrics!
          </p>
        </div>
        <div className="flex gap-3 pt-2">
          <button
            onClick={() => navigate('/best-model')}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold hover:opacity-90 transition-opacity shadow-sm"
          >
            Explore Best Model (SOTA)
          </button>
          <button
            onClick={() => navigate('/upload')}
            className="px-4 py-2 border border-border text-foreground rounded-lg text-sm font-medium hover:bg-secondary transition-colors"
          >
            Run Standard Pipeline
          </button>
        </div>
      </div>
    );
  }

  const pieData = [
    { name: 'Pass', value: overview.total_samples - overview.total_failures },
    { name: 'Fail', value: overview.total_failures },
  ];

  const radarData = [
    { metric: 'Accuracy', value: overview.cv_accuracy * 100 },
    { metric: 'Precision', value: overview.cv_precision * 100 },
    { metric: 'Recall', value: overview.cv_recall * 100 },
    { metric: 'F1 Score', value: overview.cv_f1 * 100 },
    { metric: 'AUC-ROC', value: overview.cv_auc_roc * 100 },
  ];

  const corrData = correlations.map(c => ({
    name: c.feature.length > 16 ? c.feature.slice(0, 14) + '…' : c.feature,
    fullName: c.feature,
    correlation: Number(c.correlation.toFixed(4)),
    absCorr: Number(c.abs_correlation.toFixed(4)),
  }));

  const cm = modelData?.confusion_matrix;
  const importances = modelData?.feature_importances
    ? Object.entries(modelData.feature_importances).slice(0, 15).map(([k, v]) => ({
        name: k.length > 16 ? k.slice(0, 14) + '…' : k,
        fullName: k,
        importance: Number(v.toFixed(4)),
      }))
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>

      {/* KPI cards - No Gradients */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <KpiCard icon={Layers} label="Samples" value={overview.total_samples.toLocaleString()} color="bg-primary" />
        <KpiCard icon={Activity} label="Features" value={overview.features_selected} color="bg-chart-3" />
        <KpiCard icon={Target} label="Accuracy" value={`${(overview.cv_accuracy * 100).toFixed(1)}%`} color="bg-chart-2" sub="5-fold CV" />
        <KpiCard icon={TrendingUp} label="AUC-ROC" value={`${(overview.cv_auc_roc * 100).toFixed(1)}%`} color="bg-chart-1" sub="5-fold CV" />
        <KpiCard icon={AlertTriangle} label="Anomalies" value={overview.anomalies_found} color="bg-chart-4" />
        <KpiCard icon={XCircle} label="Failures" value={overview.total_failures} color="bg-destructive" sub={`${((overview.total_failures / overview.total_samples) * 100).toFixed(1)}% rate`} />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Correlation chart */}
        <div className="lg:col-span-2 bg-card rounded-xl border border-border p-5 shadow-sm">
          <h3 className="font-semibold text-foreground mb-4">Top Feature-Failure Correlations</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={corrData} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <Tooltip
                formatter={(v, name, props) => [v, props.payload.fullName]}
                contentStyle={{ borderRadius: '8px', border: '1px solid var(--color-border)', backgroundColor: 'var(--color-card)', color: 'var(--color-foreground)' }}
              />
              <Bar dataKey="correlation" fill="var(--color-primary)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart */}
        <div className="bg-card rounded-xl border border-border p-5 shadow-sm">
          <h3 className="font-semibold text-foreground mb-4">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid var(--color-border)', backgroundColor: 'var(--color-card)' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Radar chart */}
        <div className="bg-card rounded-xl border border-border p-5 shadow-sm">
          <h3 className="font-semibold text-foreground mb-4">Model Performance (5-Fold CV)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--color-border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar name="Score" dataKey="value" stroke="var(--color-primary)" fill="var(--color-primary)" fillOpacity={0.2} strokeWidth={2} />
              <Legend />
              <Tooltip formatter={(v) => `${v.toFixed(1)}%`} contentStyle={{ borderRadius: '8px', border: '1px solid var(--color-border)', backgroundColor: 'var(--color-card)' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature importance */}
        <div className="bg-card rounded-xl border border-border p-5 shadow-sm">
          <h3 className="font-semibold text-foreground mb-4">Feature Importances (Gini)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={importances} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <Tooltip
                formatter={(v, name, props) => [v, props.payload.fullName]}
                contentStyle={{ borderRadius: '8px', border: '1px solid var(--color-border)', backgroundColor: 'var(--color-card)', color: 'var(--color-foreground)' }}
              />
              <Bar dataKey="importance" fill="var(--color-chart-3)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion matrix & Classification Report */}
      {cm && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-card rounded-xl border border-border p-5 shadow-sm">
            <h3 className="font-semibold text-foreground mb-4">Confusion Matrix (5-Fold CV)</h3>
            <div className="flex justify-center">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="p-3 text-xs text-muted-foreground"></th>
                    <th className="p-3 text-xs font-semibold text-center">Predicted PASS</th>
                    <th className="p-3 text-xs font-semibold text-center">Predicted FAIL</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="p-3 text-xs font-semibold">Actual PASS</td>
                    <td className="p-3 text-center font-bold text-lg bg-green-50/50 border border-green-200 rounded-lg min-w-[80px]">{cm[0][0]}</td>
                    <td className="p-3 text-center font-bold text-lg bg-red-50/50 border border-red-200 rounded-lg min-w-[80px]">{cm[0][1]}</td>
                  </tr>
                  <tr>
                    <td className="p-3 text-xs font-semibold">Actual FAIL</td>
                    <td className="p-3 text-center font-bold text-lg bg-orange-50/50 border border-orange-200 rounded-lg min-w-[80px]">{cm[1][0]}</td>
                    <td className="p-3 text-center font-bold text-lg bg-green-50/50 border border-green-200 rounded-lg min-w-[80px]">{cm[1][1]}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
            <div 
              className="p-5 border-b border-border bg-muted/30 flex justify-between items-center cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => setIsReportOpen(!isReportOpen)}
            >
              <div>
                <h3 className="font-semibold text-foreground">Classification Report</h3>
                <p className="text-xs text-muted-foreground mt-1">Detailed precision, recall, and F1-scores per class</p>
              </div>
              <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform ${isReportOpen ? 'rotate-180' : ''}`} />
            </div>
            
            {isReportOpen && (
              <div className="p-0 animate-in slide-in-from-top-2">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-muted-foreground bg-muted/50 uppercase border-b border-border">
                    <tr>
                      <th className="px-6 py-3 font-semibold">Class</th>
                      <th className="px-6 py-3 font-semibold">Precision</th>
                      <th className="px-6 py-3 font-semibold">Recall</th>
                      <th className="px-6 py-3 font-semibold">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border hover:bg-muted/30">
                      <td className="px-6 py-4 font-medium">PASS</td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['PASS']?.precision || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['PASS']?.recall || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['PASS']?.['f1-score'] || 0).toFixed(4)}</td>
                    </tr>
                    <tr className="border-b border-border hover:bg-muted/30">
                      <td className="px-6 py-4 font-medium flex items-center gap-2">FAIL <span className="w-2 h-2 rounded-full bg-destructive"></span></td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['FAIL']?.precision || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['FAIL']?.recall || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(modelData.classification_report['FAIL']?.['f1-score'] || 0).toFixed(4)}</td>
                    </tr>
                    <tr className="bg-primary/5 font-semibold">
                      <td className="px-6 py-4 text-primary">Accuracy</td>
                      <td colSpan="2" className="px-6 py-4"></td>
                      <td className="px-6 py-4 font-mono text-primary">{(modelData.cv_accuracy || 0).toFixed(4)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Multi-Model Comparison */}
      {modelData && modelData.models_comparison && (
        <div className="bg-card rounded-xl border border-border overflow-hidden mt-4 shadow-sm">
            <div className="p-5 border-b border-border bg-muted/30">
              <h3 className="font-semibold text-foreground">Multi-Model Comparison (SMOTE Enabled)</h3>
              <p className="text-xs text-muted-foreground mt-1">Cross-validated metrics for all evaluated models. The pipeline automatically selects the best model based on F1-Score: <strong>{modelData.best_model}</strong></p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-muted-foreground bg-muted/50 uppercase border-b border-border">
                  <tr>
                    <th className="px-6 py-3 font-semibold">Model</th>
                    <th className="px-6 py-3 font-semibold">F1-Score</th>
                    <th className="px-6 py-3 font-semibold">Recall</th>
                    <th className="px-6 py-3 font-semibold">Precision</th>
                    <th className="px-6 py-3 font-semibold">Accuracy</th>
                    <th className="px-6 py-3 font-semibold">AUC-ROC</th>
                  </tr>
                </thead>
                <tbody>
                  {[...modelData.models_comparison].sort((a, b) => b.f1 - a.f1).map((m, i) => (
                    <tr key={i} className={`border-b border-border hover:bg-muted/30 ${m.model === modelData.best_model ? 'bg-primary/5' : ''}`}>
                      <td className="px-6 py-4 font-medium flex items-center gap-2">
                        {m.model}
                        {m.model === modelData.best_model && <span className="text-[10px] bg-primary text-primary-foreground px-2 py-0.5 rounded-full uppercase tracking-wider font-bold">Selected</span>}
                      </td>
                      <td className="px-6 py-4 font-mono font-semibold text-primary">{(m.f1 || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(m.recall || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(m.precision || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(m.accuracy || 0).toFixed(4)}</td>
                      <td className="px-6 py-4 font-mono">{(m.auc_roc || 0).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        </div>
      )}
    </div>
  );
}
