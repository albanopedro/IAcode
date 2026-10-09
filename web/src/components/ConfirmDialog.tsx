import { useEffect, useRef, useState } from "react";
import type { ConfirmRequest } from "../lib/protocol";

const RISK_LABEL: Record<ConfirmRequest["risk"], string> = {
  safe: "baixo",
  sensitive: "acessa dados ou a internet",
  dangerous: "executa código",
};

interface Props {
  request: ConfirmRequest;
  onAnswer: (approved: boolean) => void;
}

/** A tool wants to run: show exactly what it will do and let the user decide. */
export function ConfirmDialog({ request, onAnswer }: Props) {
  const [left, setLeft] = useState(request.timeout);
  const denyRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    denyRef.current?.focus(); // the safe choice has the focus
    const timer = setInterval(() => setLeft((s) => Math.max(s - 1, 0)), 1000);
    return () => clearInterval(timer);
  }, [request.id]);

  return (
    <div className="modal-backdrop" role="presentation">
      <div
        className={`modal risk-${request.risk}`}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-description"
        onKeyDown={(e) => e.key === "Escape" && onAnswer(false)}
      >
        <h2 id="confirm-title">O JARVIS quer usar: {request.title}</h2>
        <pre id="confirm-description">{request.description}</pre>
        <p className="reason">
          Pede confirmação porque {request.reason}. Risco: {RISK_LABEL[request.risk]}.
        </p>
        <div className="modal-actions">
          <button type="button" ref={denyRef} className="secondary" onClick={() => onAnswer(false)}>
            Negar
          </button>
          <button type="button" className="approve" onClick={() => onAnswer(true)}>
            Permitir uma vez
          </button>
        </div>
        <p className="countdown">Sem resposta em {left}s, a ação é negada.</p>
      </div>
    </div>
  );
}
