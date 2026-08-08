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
  /** Page-in-page push into a detail layer: the incoming layer moves from
   * its navigation origin while the covered list recedes without unmounting. */
  push: {
    duration: 0.3,
    ease: [0.22, 1, 0.36, 1],
  },
  /** Return leg of the push: slightly faster so dismissal follows intent. */
  pushExit: {
    duration: 0.25,
    ease: [0.22, 1, 0.36, 1],
  },
} as const
