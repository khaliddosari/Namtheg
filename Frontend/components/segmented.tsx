"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

/**
 * A two-to-four option view switch: the shadcn toggle group as one joined
 * control, with the pressed option tinted in the accent. Pressing the current
 * option again keeps it, so there is always exactly one selected. An option
 * with a `short` label shows it on phones; its full label stays its name.
 */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  disabled,
  className,
  itemClassName,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string; short?: string }[];
  label: string;
  disabled?: boolean;
  className?: string;
  itemClassName?: string;
}) {
  return (
    <ToggleGroup
      aria-label={label}
      value={[value]}
      onValueChange={(next) => next[0] && onChange(next[0] as T)}
      variant="outline"
      size="sm"
      spacing={0}
      disabled={disabled}
      className={className}
    >
      {options.map((o) => (
        <ToggleGroupItem
          key={o.value}
          value={o.value}
          aria-label={o.short ? o.label : undefined}
          className={cn("h-9 data-pressed:bg-primary/15 data-pressed:text-primary lg:h-7", itemClassName)}
        >
          {o.short ? (
            <>
              <span aria-hidden="true" className="sm:hidden">
                {o.short}
              </span>
              <span aria-hidden="true" className="hidden sm:inline">
                {o.label}
              </span>
            </>
          ) : (
            o.label
          )}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
