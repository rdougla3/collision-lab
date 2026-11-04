import type { World } from './types.js';
import type { ModelArtifact } from './network.js';

export interface Artifact {
  id: string;
  kind: 'dataset' | 'balance' | 'model';
  name: string;
  createdAt: string;
  shared?: boolean;
  datasetId?: string;
  details: {
    counts?: { episodes: number; decisions: number; collisions: number; lookaheadCollisions: number };
    world?: World;
    network?: ModelArtifact['network'];
    bestEpoch?: number;
    bestValidationLoss?: number;
    selectedTrainTargets?: number;
    achievedCollisionFraction?: number;
    datasetFingerprint?: string;
  };
}
export interface Job {
  id: string;
  kind: string;
  status: 'preparing' | 'running' | 'complete' | 'failed' | 'cancelled';
  log: string;
  datasetId?: string;
  result?: { artifact: Artifact; files: string[] };
}
export interface ServerState { examples: Artifact[]; jobs: Job[]; busy: boolean }
