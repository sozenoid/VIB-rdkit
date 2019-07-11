#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:57:27 2019

@author: macenrola
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 

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
# =============================================================================
# 		diag.extend([1.0/(dic_weights[ats])]*3) # those masses are atomic masses so the factor 1.66e-27
# =============================================================================
		diag.extend([1.0/dic_weights[ats]/1.66e-27]*3) # those masses are atomic masses so the factor 1.66e-27

	mass_matrix = np.diag(diag)

	sqrt_mass_matrix=np.diag([x**.5 for x in diag])

	return mass_matrix, sqrt_mass_matrix

def compute_hessian(mol):
	"""
	PRE: Takes in a rdkit mol 
	POST: returns its hessian, mass weighted. The energies are given as kcal/mol and displacements in Angstrom. 
	I think this is good for numerical stability, the results should be scaled appropriately after
	"""
	dx=0.0001
	dy=dx
	print Chem.MolToMolBlock(mol)
	atom_list, atom_coords = get_xyz_from_mol(mol)
	print atom_list
	ORIG_MOL_BLOCK = Chem.MolToMolBlock(mol)

	print replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.array([[x+1 for x in y] for y in atom_coords]))

	lin_coords = np.reshape(atom_coords, (1, len(atom_coords)*3))[0]
	hessian = np.zeros((len(lin_coords), len(lin_coords)))
# =============================================================================
# 	The elements are defined as :
# 			Ex1x2   = E(x+dx, y+dy) 
# 			Ex_1x2  = E(x-dx, y+dy)
# 			Ex1x_2  = E(x+dx, y-dy)
# 			Ex_1x_2 = E(x-dx, y-dy)
# 	and the hessian ddE/dxdy = ([E(x+dx, y+dy) - E(x+dx, y-dy)]/dy - [E(x-dx, y+dy) - E(x-dx, y-dy)]/dy )/dx 
# =============================================================================
	for i in range(len(lin_coords)):
		for j in range(len(lin_coords)):
			Ex1x2_coords, Ex_1x2_coords, Ex1x_2_coords, Ex_1x_2_coords = [lin_coords.copy(), lin_coords.copy(), lin_coords.copy(), lin_coords.copy()]
			
			Ex1x2_coords[i]+=dx
			Ex1x2_coords[j]+=dy
			Ex_1x2_coords[i]-=dx
			Ex_1x2_coords[j]+=dy
			Ex1x_2_coords[i]+=dx
			Ex1x_2_coords[j]-=dy
			Ex_1x_2_coords[i]-=dx
			Ex_1x_2_coords[j]-=dy

			shape = (len(lin_coords)/3,3)
# =============================================================================
# 			print get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x2_coords, shape))) 
# 			print get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x_2_coords, shape)))
# 			print get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x2_coords, shape))) 
# 			print get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x_2_coords, shape)))
# =============================================================================
			Ex1x2   = get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x2_coords, shape)))
			Ex_1x2  = get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x2_coords, shape)))
			Ex1x_2  = get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x_2_coords, shape)))
			Ex_1x_2 = get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x_2_coords, shape)))
			ddE_dxdy = ((Ex1x2-Ex_1x2)/dy-(Ex1x_2-Ex_1x_2)/dy)/dx
# =============================================================================
# 			ddE_dxdy = ( 	
# 				(get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x2_coords, shape))) - get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex1x_2_coords, shape))))/dx  -
# 					(get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x2_coords, shape))) - get_energy(replace_coords_in_mol_block(ORIG_MOL_BLOCK, np.reshape(Ex_1x_2_coords, shape))))/dx 
# 			   )/dx
# =============================================================================
			hessian[i,j] = ddE_dxdy
			
	massmat = np.array(make_mass_matrix_from_atom_list(atom_list)[1])
	print massmat
	avg_hessian = (hessian + hessian.T)/2 *4185/1e-20/6.022e23
	print avg_hessian
	mass_weighted = np.matmul(np.matmul(massmat, avg_hessian), massmat)
	return mass_weighted

if __name__ == "__main__":
	test_mol = Chem.MolFromSmiles('C')
	test_mol = Chem.AddHs(test_mol)
	AllChem.EmbedMolecule(test_mol)
	n_steps=1000000
	tol=1e-10
	AllChem.MMFFSanitizeMolecule(test_mol)
	ff = AllChem.MMFFGetMoleculeForceField(test_mol, pyMMFFMolProperties=AllChem.MMFFGetMoleculeProperties(test_mol, mmffVariant='MMFF94', mmffVerbosity = 0), ignoreInterfragInteractions=False, nonBondedThresh=100.0)
	ff.Initialize()
	cf = ff.Minimize(n_steps, tol, tol)	
# =============================================================================
# 	print sorted([float('{0:4.4f}'.format(np.sqrt(x/5.892e-5))) for x in np.linalg.eig(np.array(compute_hessian(test_mol)))[0]])
# =============================================================================
	print sorted([float('{0:4.4f}'.format(np.sqrt(x)/2/np.pi/3e10)) for x in np.linalg.eig(np.array(compute_hessian(test_mol)))[0]])
