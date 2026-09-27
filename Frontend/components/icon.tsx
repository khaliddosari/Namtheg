import React from "react";
import { cn } from "@/lib/utils";

/**
 * Single icon surface for the app. The design system mandates Font Awesome 6.5.1
 * (no Material Symbols, no emoji), so this component maps the Material-Symbol
 * names the codebase used onto Font Awesome classes. Sizing still works via the
 * usual `style={{ fontSize }}` / text-size classes since FA glyphs are font-based.
 *
 * Usage:  <Icon name="rocket_launch" className="text-primary" style={{ fontSize: 20 }} />
 */
const MAP: Record<string, string> = {
  // navigation / chrome
  close: "fa-xmark",
  menu: "fa-bars",
  add: "fa-plus",
  arrow_forward: "fa-arrow-right",
  arrow_outward: "fa-arrow-up-right-from-square",
  open_in_new: "fa-arrow-up-right-from-square",
  expand_more: "fa-chevron-down",
  expand_less: "fa-chevron-up",
  database: "fa-database",
  build: "fa-screwdriver-wrench",
  search: "fa-magnifying-glass",
  code: "fa-code",
  check: "fa-check",
  share: "fa-share-nodes",
  help_outline: "fa-circle-question",
  tag: "fa-hashtag",
  text_fields: "fa-font",
  refresh: "fa-rotate-right",
  restart_alt: "fa-rotate-left",
  sync: "fa-arrows-rotate",

  // upload / files
  cloud_upload: "fa-cloud-arrow-up",
  upload_file: "fa-file-arrow-up",
  description: "fa-file-csv",
  delete: "fa-trash",
  download: "fa-download",
  table_view: "fa-table",
  deployed_code: "fa-cube",

  // status
  check_circle: "fa-circle-check",
  cloud_done: "fa-circle-check",
  error: "fa-circle-exclamation",
  error_outline: "fa-circle-exclamation",
  warning: "fa-triangle-exclamation",
  hourglass_empty: "fa-hourglass-half",
  health_and_safety: "fa-shield-heart",
  cloud_off: "fa-plug-circle-xmark",

  // model / data / analytics
  graph_3: "fa-diagram-project",
  model_training: "fa-gears",
  analytics: "fa-chart-column",
  insights: "fa-chart-line",
  bar_chart: "fa-chart-column",
  category: "fa-layer-group",
  target: "fa-bullseye",
  science: "fa-flask",
  bolt: "fa-bolt",
  rocket_launch: "fa-rocket",
  auto_awesome: "fa-wand-magic-sparkles",
  workspace_premium: "fa-award",
  star: "fa-star",
  zoom_in: "fa-magnifying-glass-plus",
  send: "fa-paper-plane",
  play_arrow: "fa-play",
  trending_up: "fa-arrow-trend-up",
  trending_down: "fa-arrow-trend-down",
  horizontal_rule: "fa-minus",
  image: "fa-image",
  picture_as_pdf: "fa-file-pdf",
  developer_board: "fa-microchip",
};

export function Icon({
  name,
  className,
  style,
  title,
}: {
  name: string;
  className?: string;
  style?: React.CSSProperties;
  title?: string;
}) {
  const fa = MAP[name] ?? "fa-circle";
  return (
    <i
      aria-hidden="true"
      title={title}
      className={cn("fa-solid", fa, "inline-block leading-none align-middle", className)}
      style={style}
    />
  );
}
