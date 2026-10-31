import { AiTierSettings } from "@/components/AiTierSettings";
import { PageHeader } from "@/components/ui";

export default function SettingsPage() {
  return (
    <div className="stagger mx-auto flex max-w-3xl flex-col gap-8">
      <PageHeader
        title="Settings"
        description="Workspace-wide. Changing the AI model needs the Org Admin role."
      />
      <AiTierSettings />
    </div>
  );
}
