"""Procedural 'photographs' used when no real source image is supplied.

Each generator returns (L, alpha, part) at the requested size: a grayscale luminance render,
its matte and (optionally) the mask of the single part that defines the subject (used by the
selective-recolour accent method). They are saved to sources/ and then go through
prep_subject.py like any other supplied image.
"""
