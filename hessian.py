#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:57:27 2019

@author: macenrola
"""


def get_weight_dic():
	"""
	PRE: -
	POST: Returns a dictionary with the atomic weights
	"""
	return {'C':12, 'H':1, 'N':14, 'O':16}

def get_xyz_from_mol(mol):
		"""
		PRE: Takes in a mol rdkit that has one conformer (or at least the coordinats will be taken from the first one)
		POST: returns an ndarray with the xyz coordinates
		"""
		atom_coords=[]
		atom_list = []
		conf=mol.GetConformer(0)
		for i in range(mol.GetNumAtoms()):
			atom_list.append(mol.GetAtomWithIdx(i).GetSymbol())
			atom_coords.append(np.array(list(conf.GetAtomPosition(i))))
		return atom_list, atom_coords


def replace_coords_in_mol_block(RDKIT_BLOCK_IN, coord_matrix):
	"""
	PRE: Takes in a rdkit molblock and an array with new coordinates for the said molecule with the atoms properly ordered
	POST: returns a valid molblock edited as to contain the new coordinates
	"""
	RDKIT_BLOCK = [x+'\n' for x in RDKIT_BLOCK_IN.split('\n')]
	atm_number = int(RDKIT_BLOCK[3][:3])
	for i in range(0,atm_number):
		j = i+4
		RDKIT_BLOCK[j] = RDKIT_BLOCK[j].split()
		RDKIT_BLOCK[j][:3] = coord_matrix[i, :]
		RDKIT_BLOCK[j] = (' '*(3+int(np.sign(RDKIT_BLOCK[j][0])>=0)) + '{0:.4f}'.format(RDKIT_BLOCK[j][0])+
				' '*(3+int(np.sign(RDKIT_BLOCK[j][1])>=0)) + '{0:.4f}'.format(RDKIT_BLOCK[j][1])+
				' '*(3+int(np.sign(RDKIT_BLOCK[j][2])>=0)) + '{0:.4f}'.format(RDKIT_BLOCK[j][2])+
				' {}   '.format(RDKIT_BLOCK[j][3]) + '  '.join(RDKIT_BLOCK[j][4:]) + '\n'
				)

	RDKIT_BLOCK_OUT = ''.join(RDKIT_BLOCK)
# =============================================================================
# 	print RDKIT_BLOCK_OUT
# =============================================================================
	return RDKIT_BLOCK_OUT	

def get_energy(rdkit_mol_block):
	"""
	PRE: Takes in a molecular block 
	POST: Returns a MMFF94 energy 
	"""
	tmp_mol = Chem.MolFromMolBlock(rdkit_mol_block, removeHs=False)
	AllChem.MMFFSanitizeMolecule(tmp_mol)
	ff = AllChem.MMFFGetMoleculeForceField(tmp_mol, pyMMFFMolProperties=AllChem.MMFFGetMoleculeProperties(tmp_mol, mmffVariant='MMFF94', mmffVerbosity = 0), ignoreInterfragInteractions=False, nonBondedThresh=100.0)
	ff.Initialize()
	return ff.CalcEnergy()



def make_mass_matrix_from_atom_list(atom_list):
	"""
	PRE: Takes in an atom list
	POST: Will return the mass matrix associated with it
	the matrix is square and has 3n*3n elements and is diagonal for n atoms, the square root of the matrix is also returned
	"""
	dic_weights={'C':12,'H':1,'N':14, 'O':16}
	diag=[]
	for ats in atom_list:
		diag.extend([1.0/dic_weights[ats]]*3) # those masses are atomic masses so the factor 1.66e-27

	mass_matrix = np.diag(diag)

	sqrt_mass_matrix=np.diag([x**.5 for x in diag])

	return mass_matrix, sqrt_mass_matrix

def compute_hessian(mol):
	"""
	PRE: Takes in a rdkit mol 
	POST: returns its hessian, mass weighted. The energies are given as kcal/mol and displacements in Angstrom. 
	I think this is good for numerical stability, the results should be scaled appropriately after
	"""
	
	atom_list, atom_coords = get_xyz_from_mol(mol)
	print atom_list, Chem.MolToSmiles(mol)
	ORIG_MOL_BLOCK = Chem.MolToMolBlock(mol)

	lin_coords = np.reshape(atom_coords, (1, len(atom_coords)*3))[0]
	shape = (len(lin_coords)/3,3)
	def get_energy_from_coords(coords, shape, orig_rdkit_block):
		"""
		PRE: takes in coordinates as 1d array, shape, orinigal mol block, coordinates atom number has the same order as the rdkit mol block
		POST: Returns the energy
		"""
		return get_energy(replace_coords_in_mol_block(orig_rdkit_block, np.reshape(coords, shape))) 
	
	hessian = nd.Hessian(get_energy_from_coords)(lin_coords, shape, ORIG_MOL_BLOCK)
	massmat, massmat_sqr = make_mass_matrix_from_atom_list(atom_list)
	avg_hessian = (hessian + hessian.T)/2 *4185*1e8/1e-10/6.022e23          # Units conversion factors taken from http://openmopac.net/manual/Hessian_Matrix.html
	mass_weighted = np.matmul(np.matmul(massmat_sqr, avg_hessian), massmat_sqr)
	return mass_weighted

if __name__ == "__main__":
	import rdkit
	from rdkit import Chem
	from rdkit.Chem import AllChem
	import numpy as np
	import numdifftools as nd 
	
	test_mol = Chem.MolFromSmiles('N')
	test_mol = Chem.AddHs(test_mol)
	AllChem.EmbedMolecule(test_mol)
	print "BEFORE OPTI"
	print Chem.MolToMolBlock(test_mol)
	n_steps=1000000
	tol=1e-10
	AllChem.MMFFSanitizeMolecule(test_mol)
	ff = AllChem.MMFFGetMoleculeForceField(test_mol, pyMMFFMolProperties=AllChem.MMFFGetMoleculeProperties(test_mol, mmffVariant='MMFF94', mmffVerbosity = 0), ignoreInterfragInteractions=False, nonBondedThresh=100.0)
	ff.Initialize()
	cf = ff.Minimize(n_steps, tol, tol)	
	print "AFTER OPTI ({})".format(cf)
	print Chem.MolToMolBlock(test_mol)
	print [float('{0:4.4f}'.format(np.sqrt(x*6.022e28)/2/np.pi/2.99e10)) for x in sorted(np.linalg.eig(np.array(compute_hessian(test_mol)))[0])] # Conversion factors taken from http://openmopac.net/manual/Hessian_Matrix.html
