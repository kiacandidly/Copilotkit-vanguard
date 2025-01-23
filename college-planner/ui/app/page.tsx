"use client";

import dynamic from "next/dynamic";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TripsProvider } from "@/lib/hooks/use-trips";
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

// Disable server-side rendering for the MapCanvas component, this
// is because Leaflet is not compatible with server-side rendering
//
// https://github.com/PaulLeCam/react-leaflet/issues/45
let MapCanvas: any;
MapCanvas = dynamic(
  () =>
    import("@/components/MapCanvas").then((module: any) => module.MapCanvas),
  {
    ssr: false,
  }
);

export default function Home() {
  const lgcDeploymentUrl =
    globalThis.window === undefined
      ? null
      : new URL(window.location.href).searchParams.get("lgcDeploymentUrl");
  return (
    <CopilotKit
      agent="college_planner"
      runtimeUrl={
        process.env.NEXT_PUBLIC_CPK_PUBLIC_API_KEY == undefined
          ? `/api/copilotkit?lgcDeploymentUrl=${lgcDeploymentUrl ?? ""}`
          : "https://api.cloud.copilotkit.ai/copilotkit/v1"
      }
      publicApiKey={process.env.NEXT_PUBLIC_CPK_PUBLIC_API_KEY}
    >
      <CopilotSidebar
        defaultOpen={false}
        clickOutsideToClose={false}
        labels={{
          title: "College planner",
          initial:
            "Hi! 👋 How can I help you plan for college?",
        }}
      >
        <TooltipProvider>
          <TripsProvider>
            <main className="h-screen w-screen flex">
              <div className="w-2/3 h-full">
                <MapCanvas />
              </div>
            </main>
          </TripsProvider>
        </TooltipProvider>
      </CopilotSidebar>
    </CopilotKit>
  );
}
