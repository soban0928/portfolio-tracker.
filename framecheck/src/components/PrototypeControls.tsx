import { useId } from 'react';
import { MOCK_SCENARIO_LABELS, type MockScenarioId } from '../services/mock/mockAnalysisFixtures';
import type { MockScenarioChoice } from '../services/mock/mockAnalysisService';

interface PrototypeControlsProps {
  scenario: MockScenarioChoice;
  onChange: (scenario: MockScenarioChoice) => void;
  disabled: boolean;
}

/** Lets reviewers preview each mock result and the error state. Prototype only. */
export function PrototypeControls({ scenario, onChange, disabled }: PrototypeControlsProps) {
  const selectId = useId();
  return (
    <details className="proto-controls">
      <summary>Prototype settings</summary>
      <div className="proto-controls__body">
        <label htmlFor={selectId}>Sample result to show</label>
        <select
          id={selectId}
          value={scenario}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value as MockScenarioChoice)}
        >
          <option value="auto">Automatic (varies by file)</option>
          {(Object.keys(MOCK_SCENARIO_LABELS) as MockScenarioId[]).map((id) => (
            <option key={id} value={id}>
              {MOCK_SCENARIO_LABELS[id]}
            </option>
          ))}
          <option value="analysis_error">Simulate an analysis error</option>
        </select>
        <p className="muted">All options use illustrative mock data. Nothing is analyzed.</p>
      </div>
    </details>
  );
}
