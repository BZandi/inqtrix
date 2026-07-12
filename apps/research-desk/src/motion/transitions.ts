export const appMotion = {
  card: {
    duration: 0.22,
    ease: [0.22, 1, 0.36, 1],
  },
  composer: {
    duration: 0.16,
    ease: [0.22, 1, 0.36, 1],
  },
  list: {
    duration: 0.18,
    ease: [0.22, 1, 0.36, 1],
  },
  panel: {
    duration: 0.26,
    ease: [0.22, 1, 0.36, 1],
  },
  /** Page-in-page push (drill into a detail layer): the incoming layer
   * slides in full width while the covered list parallaxes to -30%
   * (Apple navigation structure on the house curve). */
  push: {
    duration: 0.3,
    ease: [0.22, 1, 0.36, 1],
  },
  /** Back leg of the push, one step faster (Fluent exit convention). */
  pushExit: {
    duration: 0.25,
    ease: [0.22, 1, 0.36, 1],
  },
} as const
