import type { SectionDef } from "@/types/curriculum";
import { week1, week2, week3, week4 } from "./weeks01-04";
import { week5, week6, week7, week8 } from "./weeks05-08";
import { week9, week10, week11, week12 } from "./weeks09-12";

/**
 * The twelve weeks of the IITM "Machine Learning Techniques" course, in course order.
 * They are sections of the Linear Algebra module (part "mlt"), not a separate course.
 */
export const mltWeeks: SectionDef[] = [week1, week2, week3, week4, week5, week6, week7, week8, week9, week10, week11, week12];
